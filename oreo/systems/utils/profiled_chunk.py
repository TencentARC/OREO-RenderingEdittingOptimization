"""
Profile-based Adaptive chunk scheduler.

Core classes:
- ProfiledScheduler: generator, automatic profiling Video memory consumption and dynamic calculation chunk_size

Principle:
1. first time yield Return probe chunk(Small), the caller returns control after processing
2. Measure probe Peak memory increment during → Calculate bytes_per_item
3. According to the remaining video memory and bytes_per_item Calculate optimal chunk_size
4. Follow-up yield Use the best chunk_size

Usage:
    # cat merge(normal Chunked)
    results = []
    for start, size in ProfiledScheduler(K, device, probe_size=2000):
        results.append(compute(data[start:start+size]))
    output = torch.cat(results, dim=0)

    # z-buffer merge(rast Chunked)
    for start, size in ProfiledScheduler(num_faces, device, probe_size=50000):
        rast = rasterize(faces[start:start+size])
        zbuffer_merge(...)
"""

import logging
import torch


class ProfiledScheduler:
    """
    Profile-based Adaptive chunk Scheduler (generator).

    take advantage of Python generator of yield pause/Recovery mechanism:
    - first time yield After that, the caller handles probe chunk
    - Measure the video memory when control returns and calculate the optimal chunk_size
    - Follow-up yield Use the best chunk_size

    chunk_size Calculation formula::

        bytes_per_item = probe_delta / probe_size × safety_factor
        available      = (GPUtotal video memory - Currently occupied) × target_usage
        chunk_size     = clamp(available / bytes_per_item, min_chunk, max_chunk)

    Three lines of defense:
    1. safety_factor — Conservatism (overestimation) at the unit estimate level bytes_per_item → underestimate chunk_size)
    2. target_usage  — Global memory level conservation (reserved for allocator fragments / other processes)
    3. min/max_chunk — Hard clamp pocket bottom

    Args:
        total: total item Quantity.
        device: CUDA equipment.
        probe_size: probe size (first chunk size).
            - too small → Video memory delta Be CUDA allocator Alignment/Submerged in debris noise, estimation is inaccurate.
            - too big → When memory is tight probe itself OOM.
            - Experience: Take expectations chunk_size of 1/10 ~ 1/50.
            - Recalculation scenarios (such as 26-neighbor normal) use small values (~2000);
              GPU Parallel scenarios (such as nvdiffrast rasterize) with a large value (~50000) to obtain a stable signal.
        safety_factor: safety factor, multiplied by bytes_per_item On.
            - 1.3(30% margin): suitable for scenarios with even overhead (such as voxel normal calculation).
            - 1.5(50% margin): suitable for scenarios with large fluctuations (such as nvdiffrast internal buffer pre-allocated,
              The density of the dough is uneven).
        target_usage: Target video memory utilization (0~1), default 0.85.
            - Reserve 15% give PyTorch allocator Internal fragmentation and other processes.
            - with safety_factor Form double insurance.
        min_chunk: smallest chunk size.
            - To prevent memory shortage chunk_size Too small will cause the number of cycles to explode,Python loop overhead Dominate.
            - Recalculation scenario ~500;GPU Parallel scenario ~10000.
        max_chunk: maximum chunk size.
            - prevent probe underestimate bytes_per_item lead to next chunk OOM.
            - nvdiffrast Hard limit: single rasterize ≤ 2^24(~1677Thousands) noodles,
              rast Scenario suggestions max_chunk ≤ 2000000.
    """

    def __init__(
        self,
        total: int,
        device: torch.device,
        probe_size: int = 2000,
        safety_factor: float = 1.3,
        target_usage: float = 0.85,
        min_chunk: int = 500,
        max_chunk: int = 500000,
    ):
        self.total = total
        self.device = device
        self.probe_size = min(probe_size, total)
        self.safety_factor = safety_factor
        self.target_usage = target_usage
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk
        # profiling Result (externally readable)
        self.bytes_per_item: float = 0.0
        self.chunk_size: int = 0

    def __iter__(self):
        dev_idx = self.device.index if self.device.index is not None else 0
        total = self.total
        probe_n = self.probe_size

        # total amount ≤ probe_size, done in one go
        if total <= probe_n:
            yield (0, total)
            return

        # ---- Phase 1: Probe ----
        torch.cuda.reset_peak_memory_stats(dev_idx)
        mem_before = torch.cuda.max_memory_allocated(dev_idx)

        yield (0, probe_n)  # Pause, caller handles probe chunk

        # Recovery:probe Processed, measure memory
        mem_after = torch.cuda.max_memory_allocated(dev_idx)
        delta = max(mem_after - mem_before, 1)
        self.bytes_per_item = delta / probe_n * self.safety_factor

        # Calculate optimal chunk_size
        total_mem = torch.cuda.get_device_properties(dev_idx).total_memory
        allocated_now = torch.cuda.memory_allocated(dev_idx)
        available = int((total_mem - allocated_now) * self.target_usage)
        optimal = int(available / self.bytes_per_item) if self.bytes_per_item > 0 else self.max_chunk
        self.chunk_size = max(self.min_chunk, min(optimal, self.max_chunk))

        logging.debug(
            f"[ProfiledScheduler] probe={probe_n}, "
            f"delta={delta / 1024 / 1024:.1f}MB, "
            f"bytes/item={self.bytes_per_item:.0f}, "
            f"chunk_size={self.chunk_size}, total={total}"
        )

        # ---- Phase 2: optimal chunking ----
        offset = probe_n
        while offset < total:
            size = min(self.chunk_size, total - offset)
            yield (offset, size)
            offset += size
