"""True-async pipeline-parallel helpers for overlapping Trellis train with FlowEdit guidance."""

import logging
import threading
import time
from collections import deque
from typing import List, Any, Callable, TypeVar, Generic, Dict, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from PIL import Image

from oreo.guidance.base import GuidanceResult


T = TypeVar('T')


# =====================================================================
# AsyncGuidanceResult — asynchronous Guidance Return results
# =====================================================================

@dataclass
class AsyncGuidanceResult:
    """Async guidance handle (future / events) vs eager GuidanceResult."""
    rgb_grad: torch.Tensor                                  # (B, V, H, W, 3)
    loss_scalar: float = 0.0                                # weighted scalar loss
    loss_dict: Optional[Dict[str, float]] = None            # Segmentation loss(for logs)
    edited_imgs: Optional[torch.Tensor] = None              # (B, V, C, H, W) Edited image
    trackers: Optional[List] = None                         # FlowEdit StateTracker list
    # ---- Wall clock timing (dual GPU Utilization analysis)----
    guid_wall_start: float = 0.0                            # worker start GPU calculated perf_counter
    guid_wall_end: float = 0.0                              # worker Complete GPU calculated perf_counter

    @classmethod
    def from_guidance_result(
        cls,
        result: "GuidanceResult",
        rgb_grad: torch.Tensor,
        guidance_weight: float,
    ) -> "AsyncGuidanceResult":
        """
        from GuidanceResult + rgb_grad Build asynchronous results.

        centralized processing detach / scalar convert, avoid submit_async Scattered conversion logic internally.
        """
        loss_scalar = (result.loss * guidance_weight).item()
        loss_dict_scalar: Optional[Dict[str, float]] = None
        if result.loss_dict:
            loss_dict_scalar = {
                k: (v.item() if isinstance(v, torch.Tensor) else float(v))
                for k, v in result.loss_dict.items()
                if v is not None
            }
        return cls(
            rgb_grad=rgb_grad,
            loss_scalar=loss_scalar,
            loss_dict=loss_dict_scalar,
            edited_imgs=result.edited_imgs.detach() if result.edited_imgs is not None else None,
            trackers=result.trackers,  # internal tensor Already detached clone
        )


# =====================================================================
# PipelineParallelMixin — True asynchronous version
# =====================================================================

class PipelineParallelMixin:
    """Mixin that runs guidance forward+backward on a background thread without blocking the train step."""

    # Declare required attributes (subclasses must provide them)
    device: torch.device

    def _init_pipeline_parallel(self, num_streams: int = 2):
        """
        Initialize the pipeline parallel infrastructure.

        Subclasses need to be in __init__ Call this method in .

        Args:
            num_streams: CUDA stream quantity (default 2, double buffering)
        """
        self._pp_streams = [
            torch.cuda.Stream(device=self.device)
            for _ in range(num_streams)
        ]
        self._pp_queue: deque = deque()  # None maxlen Restrictions, supports any accum_steps
        self._pp_slot_counter = 0
        self._pp_num_streams = num_streams
        self._pp_thread: Optional[threading.Thread] = None  # Background thread reference
        self._pp_error: Optional[BaseException] = None       # Background thread exception capture

        logging.info(f"[PipelineParallel] Async mode enabled with {num_streams} streams on {self.device}")

    def _join_prev_thread(self) -> None:
        """
        Wait for the previous background thread to complete and check for exceptions.

        Calling time:
        - submit_async Beginning: guaranteed Guidance GPU Serial execution (only run one at a time guidance)
        - wait_and_get Start: Make sure the result is enqueued

        Exception propagation: Exceptions caught by the background thread will be re-thrown here and will not be swallowed silently.
        """
        if self._pp_thread is not None:
            self._pp_thread.join()
            self._pp_thread = None
        if self._pp_error is not None:
            err = self._pp_error
            self._pp_error = None
            raise RuntimeError(f"[PipelineParallel] Background guidance thread failed: {err}") from err

    def submit_async(
        self,
        comp_rgb: torch.Tensor,
        condition_images: List[Image.Image],
        *,
        guidance_weight: float = 1.0,
        guidance_cfg: Any,
        rank: int = 0,
        **kwargs,
    ) -> None:
        """
        True asynchronous submission guidance Computational tasks.

        in the background Thread executed in guidance forward + backward, the main thread returns immediately.
        Train GPU You can continue to do it during this period drain_prev / build_next Waiting for work.

        Process:
          main thread:
            0. join Previous background thread (guaranteed Guidance GPU serial)
            1. detach comp_rgb → move to Guidance GPU → requires_grad proxy
            2. Start background thread → Return immediately
          background thread:
            3. compute_guidance(proxy, ..., guidance_cfg=...) → GuidanceResult
            4. (loss * weight).backward() → proxy.grad = rgb_grad
            5. Pack AsyncGuidanceResult → Join the team

        Args:
            comp_rgb: render image (B,V,H,W,C),float [0,1], from train side (with autograd Figure)
            condition_images: Conditional image list [len=B] of PIL.Image
            guidance_weight: guidance loss weight
            guidance_cfg: Runtime configuration (per-stage), passed to compute_guidance
            rank: of the current process rank(passed to compute_guidance)
            **kwargs: passed to compute_guidance additional parameters
        """
        # ---- 0. Wait for the previous one submit done (Guidance GPU serial to avoid memory contention) ----
        self._join_prev_thread()

        # ---- 1. in Train GPU on detach → move to Guidance GPU ----
        # Sync only comp_rgb The device's current stream(Not globally synchronized to avoid blocking Guidance GPU)
        torch.cuda.current_stream(comp_rgb.device).synchronize()
        proxy_rgb = comp_rgb.detach().to(self.device).requires_grad_(True)  # (B, V, H, W, 3), in Guidance GPU on

        # ---- 2. Execute in background thread guidance forward + backward ----
        slot_idx = self._pp_slot_counter % self._pp_num_streams
        stream = self._pp_streams[slot_idx]

        def _worker():
            try:
                _wall_start = time.perf_counter()
                with torch.cuda.stream(stream):
                    # guidance forward(The most time-consuming part,~59s)
                    result: GuidanceResult = self.compute_guidance(
                        proxy_rgb, condition_images,
                        guidance_cfg=guidance_cfg, rank=rank, **kwargs
                    )
                    # guidance backward → rgb_grad
                    (result.loss * guidance_weight).backward()
                    rgb_grad = proxy_rgb.grad.detach()  # (B, V, H, W, 3)

                    # Pack results
                    async_result = AsyncGuidanceResult.from_guidance_result(
                        result, rgb_grad, guidance_weight
                    )

                    # release guidance side intermediate product
                    # ★ Don't del proxy_rgb: It is the variable captured by the closure,
                    #   del will let Python Mark it as a local variable → UnboundLocalError
                    del result

                # ensure stream All in kernel Join the team after completion
                stream.synchronize()
                _wall_end = time.perf_counter()
                async_result.guid_wall_start = _wall_start
                async_result.guid_wall_end = _wall_end
                self._pp_queue.append(async_result)
            except BaseException as e:
                self._pp_error = e

        self._pp_thread = threading.Thread(target=_worker, daemon=True)
        self._pp_thread.start()  # ★ Return now! The main thread can continue to do Train GPU work on
        self._pp_slot_counter += 1

    def wait_and_get(
        self,
        target_device: Optional[torch.device] = None,
    ) -> AsyncGuidanceResult:
        """
        Get the earliest submission submit_async Result (FIFO).

        normal case(Queue is not empty): Direct pop, no join The current background thread.
          → guid(N) Continue running in the background thread,P3(N-1) Can be executed in parallel.
        border case(The queue is empty, such as accum end / epoch end):
          join The current background thread is waiting for the result to be enqueued.

        Args:
            target_device: rgb_grad The target device (None then remain at Guidance GPU (up)

        Returns:
            AsyncGuidanceResult: contains rgb_grad + scalar log

        Raises:
            RuntimeError: Background thread exception or Queue is empty
        """
        # ---- Only blocks and waits when the queue is empty (boundary case) ----
        # normal case: submit_async(N) Already join guid(N-1) → results in queue → direct pop
        # border case: accum end/epoch There is no next one at the end submit → Need to wait manually
        if not self._pp_queue:
            self._join_prev_thread()

        if not self._pp_queue:
            raise RuntimeError("No pending async submission. Call submit_async() first.")

        # Remove from the head of the queue (FIFO)
        async_result: AsyncGuidanceResult = self._pp_queue.popleft()

        # Move to target device (if needed)
        if target_device is not None:
            if async_result.rgb_grad.device != target_device:
                async_result.rgb_grad = async_result.rgb_grad.to(target_device)
            if async_result.edited_imgs is not None and async_result.edited_imgs.device != target_device:
                async_result.edited_imgs = async_result.edited_imgs.to(target_device)
            # ★ trackers remain in guidance GPU:get_progress_grid(pipe, n) need
            #   pipe (VAE) and tracker internal latent All are there guidance GPU on

        return async_result

    def has_pending(self) -> bool:
        """Check if there is pending asynchronous submission."""
        return len(self._pp_queue) > 0 or self._pp_thread is not None

    @property
    def pending_count(self) -> int:
        """Return pending Number of tasks (including background threads being executed)."""
        count = len(self._pp_queue)
        if self._pp_thread is not None and self._pp_thread.is_alive():
            count += 1
        return count
