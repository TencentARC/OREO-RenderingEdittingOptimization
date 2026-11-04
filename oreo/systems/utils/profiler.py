"""PhaseProfiler — CUDA Event phase-level timing for OREO autograd phases."""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.distributed as dist


# =====================================================================
# PhaseProfiler — Basic timer
# =====================================================================

@dataclass
class PhaseProfiler:
    """CUDA-event phase timer for OREO autograd phases."""
    enabled: bool = True
    verbose: bool = True    # control step() Whether to print summary (multiple rank Only the main process is set True)

    # ---- Single step timing ----
    _events: Dict[str, torch.cuda.Event] = field(default_factory=dict)
    _order: List[str] = field(default_factory=list)

    # ---- Cumulative statistics ----
    _history: Dict[str, List[float]] = field(default_factory=dict)
    _step_count: int = 0

    def tick(self, name: str) -> None:
        """Record a stage boundary. two adjacent tick The time between is the time taken for this phase."""
        if not self.enabled:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._events[name] = event
        self._order.append(name)

    def elapsed(self) -> Dict[str, float]:
        """
        Return adjacent tick The time elapsed between (ms).

        required at all tick Called afterwards. will call torch.cuda.synchronize()
        ensure all GPU Read the time after the operation is complete.

        Returns:
            Dict[str, float]: {phase_name: elapsed_ms},
            last item "total" Be the first and last tick total time between.
        """
        if not self.enabled or len(self._order) < 2:
            return {}
        torch.cuda.synchronize()
        timings: Dict[str, float] = {}
        for i in range(len(self._order) - 1):
            name = self._order[i]
            start = self._events[self._order[i]]
            end = self._events[self._order[i + 1]]
            timings[name] = start.elapsed_time(end)  # ms
        timings["total"] = self._events[self._order[0]].elapsed_time(
            self._events[self._order[-1]]
        )  # ms
        return timings

    def reset(self) -> None:
        """Reset single step timing (does not clear cumulative statistics)."""
        self._events.clear()
        self._order.clear()

    def step(
        self,
        timings: Dict[str, float],
        global_step: int,
        print_freq: int = 10,
    ) -> Optional[Dict[str, float]]:
        """
        Cumulative step timing, and at each print_freq Print summary of steps.

        Args:
            timings: elapsed() Returned timing dictionary
            global_step: Current global step count
            print_freq: Printing frequency (every N (Print the average value step by step)

        Returns:
            If this step triggers printing, return the average dictionary; otherwise return None.
        """
        if not self.enabled or not timings:
            self.reset()
            return None

        # accumulate
        for k, v in timings.items():
            if k not in self._history:
                self._history[k] = []
            self._history[k].append(v)
        self._step_count += 1

        # Reset single step timing
        self.reset()

        # Check if it is time to print
        if self._step_count % print_freq != 0:
            return None

        # Calculate average
        avg: Dict[str, float] = {}
        for k, vals in self._history.items():
            avg[k] = sum(vals) / len(vals)

        # Print
        if self.verbose:
            self._print_summary(avg, global_step)

        # clear accumulation
        self._history.clear()
        self._step_count = 0

        return avg

    def _print_summary(self, avg: Dict[str, float], global_step: int) -> None:
        """Print summary (subclasses can override to extend format)."""
        n = len(next(iter(self._history.values())))
        lines = [f"[PhaseProfiler] Step {global_step} | avg over {n} steps (ms):"]

        # press key Print sequentially (total put last)
        for k in avg:
            if k != "total":
                lines.append(f"  {k:30s} {avg[k]:8.1f}")
        if "total" in avg:
            lines.append(f"  {'total':30s} {avg['total']:8.1f}")

        logging.info("\n".join(lines))

    def as_log_dict(self, timings: Dict[str, float], prefix: str = "time/") -> Dict[str, float]:
        """
        Convert the timing dictionary to a format suitable for the logging system.

        Args:
            timings: elapsed() Returned timing dictionary
            prefix: Key name prefix

        Returns:
            Dict[str, float]: {prefix+phase_name: elapsed_ms}
        """
        if not timings:
            return {}
        return {f"{prefix}{k}": v for k, v in timings.items()}

    def collect(self, global_step: int, print_freq: int = 10) -> Dict[str, float]:
        """
        One-stop collection:elapsed → merge log → cumulative summary → Reset.

        The caller only needs to use all tick adjust later collect(), get time/* Just merge the dictionary into the log.
        No need to call manually elapsed / as_log_dict / step.

        Args:
            global_step: Current global step count
            print_freq: Printing frequency (every N (Print the average value step by step)

        Returns:
            Dict[str, float]: {time/phase_name: elapsed_ms},
            enabled=False Returns an empty dictionary.
        """
        timings = self.elapsed()
        log_dict = self.as_log_dict(timings)
        self.step(timings, global_step, print_freq=print_freq)
        return log_dict


# =====================================================================
# GPUUtilizationMixin — double GPU Utilization analysis
# =====================================================================

class GPUUtilizationMixin:
    """
    for PhaseProfiler add double GPU Wall clock time tracking and utilization analysis.

    Pass time.perf_counter() record each phase 's wall clock time,
    combine Guidance GPU active period, calculate:
    - guid_gpu_active: Guidance GPU active time
    - overlap_guid_P3: guid with P3 The overlap time (= asynchronous returns)
    - train_gpu_idle: Train GPU free time
    - guid_gpu_idle: Guidance GPU free time

    Design intent:
      CUDA Event Time intervals can only be measured on the same device and cannot be compared across devices.
      Can be analyzed using wall clock time Train GPU and Guidance GPU time overlap relationship.
    """

    # ---- Needs to be provided by the composite class ----
    enabled: bool

    def _init_gpu_util(self) -> None:
        """initialization GPU Utilization tracking status. subcategory in __post_init__ call in."""
        self._wall_times: Dict[str, float] = {}
        self._guid_wall_start: float = 0.0
        self._guid_wall_end: float = 0.0

    def set_guid_timing(self, wall_start: float, wall_end: float) -> None:
        """
        settings Guidance GPU active time period.

        by worker Thread passes AsyncGuidanceResult Pass back, in drain_prev call in.

        Args:
            wall_start: guidance Calculation starts time.perf_counter() value
            wall_end:   guidance calculation ended time.perf_counter() value
        """
        if not self.enabled:
            return
        self._guid_wall_start = wall_start
        self._guid_wall_end = wall_end

    def _tick_wall(self, name: str) -> None:
        """Record the wall clock time of a phase boundary. in tick() is called."""
        self._wall_times[name] = time.perf_counter()

    def _compute_gpu_util(self, timings: Dict[str, float]) -> None:
        """
        Calculate GPU Utilization indicators, written in place timings Dictionary.

        in elapsed() is called at the end. need _wall_times There is "P3_grad_backward" and "end",
        and _guid_wall_end > 0(i.e. set_guid_timing been called).

        Args:
            timings: elapsed() calculated phase timing Dictionary (will be expanded in-place)
        """
        if not (self._guid_wall_end > 0
                and "P3_grad_backward" in self._wall_times
                and "end" in self._wall_times):
            return

        guid_s = self._guid_wall_start
        guid_e = self._guid_wall_end
        p3_s = self._wall_times["P3_grad_backward"]
        p3_e = self._wall_times["end"]

        # Guidance GPU active time (ms)
        guid_active_ms = (guid_e - guid_s) * 1000.0
        timings["guid_gpu_active"] = guid_active_ms

        # guid with P3 overlap time = true async return(ms)
        overlap_ms = max(0.0, (min(guid_e, p3_e) - max(guid_s, p3_s))) * 1000.0
        timings["overlap_guid_P3"] = overlap_ms

        # Train GPU free ≈ total - Yes GPU calculated phase sum (ms)
        train_active_phases = ["dense_sampling", "P1_rollout",
                               "P2a_decode_render", "P3_grad_backward"]
        train_active_ms = sum(timings.get(p, 0.0) for p in train_active_phases)
        timings["train_gpu_idle"] = max(0.0, timings.get("total", 0.0) - train_active_ms)

        # Guidance GPU free ≈ total wall clock time - guid active time (ms)
        # take _wall_times earliest of tick(Due to dict coverage, actually iter N the first tick)
        wall_keys = list(self._wall_times.keys())
        if wall_keys:
            first_wall = self._wall_times[wall_keys[0]]
            total_wall_ms = (self._wall_times["end"] - first_wall) * 1000.0
            timings["guid_gpu_idle"] = max(0.0, total_wall_ms - guid_active_ms)

    def _reset_gpu_util(self) -> None:
        """Reset wall clock time state. in reset() is called."""
        self._wall_times.clear()
        self._guid_wall_start = 0.0
        self._guid_wall_end = 0.0


# =====================================================================
# MemoryMixin — every stage GPU Video memory snapshot
# =====================================================================

# Memory snapshot indicator key prefix (each phase at the end allocated GiB)
_MEM_PREFIX = "mem@"


class MemoryMixin:
    """
    for PhaseProfiler add GPU Memory tracking.

    in every tick time record ``torch.cuda.memory_allocated()``,
    elapsed() When injected into the video memory at the end of each stage (``mem@{phase}``) and the peak value of this step (``mem@peak``).

    Design intent:
      with GPUUtilizationMixin parallel Mixin,AsyncPhaseProfiler Combine both at the same time.
      Video memory snapshots for each stage can be located OOM During the high water stage, the peak value can be used to monitor the overall trend.

    Key Agreement (timings dict within):
      ``mem@dense_sampling``  → dense_sampling at the end allocated GiB
      ``mem@peak``            → of this step max_memory_allocated GiB
    """

    # ---- Needs to be provided by the composite class ----
    enabled: bool
    _order: List[str]

    def _init_mem(self) -> None:
        """Initialize the memory tracking state. subcategory in __post_init__ call in."""
        self._mem_at_tick: Dict[str, float] = {}  # tick_name -> allocated GiB

    def _tick_mem(self, name: str) -> None:
        """Record a stage boundary GPU allocated video memory. in tick() is called."""
        self._mem_at_tick[name] = torch.cuda.memory_allocated() / (1024 ** 3)  # GiB

    def _compute_mem(self, timings: Dict[str, float]) -> None:
        """
        Calculate the video memory at the end of each stage + Peak, write in place timings Dictionary.

        in elapsed() is called at the end.

        written key:
          ``mem@{phase_name}`` — the phase at the end allocated Video memory (GiB)
          ``mem@peak``         — This step max_memory_allocated (GiB)

        Args:
            timings: elapsed() calculated phase timing Dictionary (will be expanded in-place)
        """
        for i in range(len(self._order) - 1):
            phase_name = self._order[i]
            end_tick = self._order[i + 1]
            if end_tick in self._mem_at_tick:
                timings[f"{_MEM_PREFIX}{phase_name}"] = self._mem_at_tick[end_tick]
        timings[f"{_MEM_PREFIX}peak"] = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
        )

    def _reset_mem(self) -> None:
        """Reset memory tracking status + Peak statistics. in reset() is called."""
        self._mem_at_tick.clear()
        torch.cuda.reset_peak_memory_stats()  # next step peak Recalculate from current value


# =====================================================================
# AsyncPhaseProfiler — Dedicated to asynchronous pipelines
# =====================================================================

# GPU Utilization indicator key Collection (for printing partitions)
_GPU_UTIL_KEYS = frozenset({
    "guid_gpu_active", "overlap_guid_P3",
    "train_gpu_idle", "guid_gpu_idle",
})


@dataclass
class AsyncPhaseProfiler(MemoryMixin, GPUUtilizationMixin, PhaseProfiler):
    """
    PhaseProfiler + double GPU Utilization analysis + Memory diagnostics for asynchronous Guidance assembly line.

    Combine three orthogonal dimensions:
    - PhaseProfiler:       CUDA Event Timing
    - GPUUtilizationMixin: double GPU wall clock time + Utilization analysis
    - MemoryMixin:         every stage GPU allocated Snapshot + peak tracking

    Video memory key Agreement:
      timings dict China and Israel ``mem@`` prefix storage (GiB),
      as_log_dict() convert to ``mem/`` prefix write CSV / wandb.

    across rank Video memory visibility:
      step() After overwriting, each rank Independent use logging.info Print a one-line video memory summary,
      cooperate _setup_file_logging of per-rank Log files can be compared afterwards rank video memory.

    Output example (rank 0,verbose=True Additional output is complete when phase Summary):
        [AsyncPhaseProfiler] Step 10 | avg over 10 steps:
          dense_sampling                     3322.3 ms  | mem 45.3 GiB
          P1_rollout                        11502.9 ms  | mem 62.1 GiB
          ...
          total                            131049.6 ms  | peak 92.6 GiB

    each rank will print a line of video memory summary (write their respective run_rank{i}.log):
        [Rank 0 Step 10] mem(GiB): dense_sampling: 45.3, ..., peak=73.2
        [Rank 2 Step 10] mem(GiB): dense_sampling: 46.1, ..., peak=92.6
    """

    def __post_init__(self):
        self._init_gpu_util()
        self._init_mem()

    def tick(self, name: str) -> None:
        super().tick(name)
        if self.enabled:
            self._tick_wall(name)
            self._tick_mem(name)

    def elapsed(self) -> Dict[str, float]:
        timings = super().elapsed()
        if self.enabled:
            self._compute_gpu_util(timings)
            self._compute_mem(timings)
        return timings

    def step(
        self,
        timings: Dict[str, float],
        global_step: int,
        print_freq: int = 10,
    ) -> Optional[Dict[str, float]]:
        """
        Overwrite:print_freq When triggered, all rank Print a separate line of video memory summary.

        Detailed phase Aggregated by _print_summary Process(verbose-gated, only rank 0);
        The video memory summary is given by _log_mem_all_ranks Process (all rank, written to respective log files).
        """
        avg = super().step(timings, global_step, print_freq)
        if avg is not None:
            self._log_mem_all_ranks(avg, global_step)
        return avg

    def reset(self) -> None:
        super().reset()
        self._reset_gpu_util()
        self._reset_mem()

    def as_log_dict(self, timings: Dict[str, float], prefix: str = "time/") -> Dict[str, float]:
        """
        Overwrite:mem@ beginning key Use ``mem/`` prefix, the rest will be used ``time/`` prefix.

        This way CSV / wandb in the log key for:
          time/dense_sampling, time/P1_rollout, ...
          mem/dense_sampling, mem/P1_rollout, ..., mem/peak
        """
        if not timings:
            return {}
        result: Dict[str, float] = {}
        for k, v in timings.items():
            if k.startswith(_MEM_PREFIX):
                # mem@phase_name -> mem/phase_name
                result[f"mem/{k[len(_MEM_PREFIX):]}"] = v
            else:
                result[f"{prefix}{k}"] = v
        return result

    def _log_mem_all_ranks(self, avg: Dict[str, float], global_step: int) -> None:
        """
        each rank Independent printing of video memory summary (via logging.info).

        cooperate _setup_file_logging for each rank Created run_rank{i}.log,
        You can compare each later rank The amount of video memory, positioning list rank OOM.
        Not used all_reduce, zero communication overhead.
        """
        rank = dist.get_rank() if dist.is_initialized() else 0

        peak_key = f"{_MEM_PREFIX}peak"
        parts: List[str] = []
        for k, v in avg.items():
            if k.startswith(_MEM_PREFIX) and k != peak_key:
                parts.append(f"{k[len(_MEM_PREFIX):]}: {v:.1f}")
        peak = avg.get(peak_key, 0.0)

        logging.info(
            f"[Rank {rank} Step {global_step}] "
            f"mem(GiB): {', '.join(parts)}, peak={peak:.1f}"
        )

    def _print_summary(self, avg: Dict[str, float], global_step: int) -> None:
        """Overlay printing: General phase + Video memory snapshot + GPU Utilization."""
        n = len(next(iter(self._history.values())))
        lines = [f"[AsyncPhaseProfiler] Step {global_step} | avg over {n} steps:"]

        # structure phase -> Video memory mapping
        mem_for_phase: Dict[str, float] = {}
        for k, v in avg.items():
            if k.startswith(_MEM_PREFIX) and k != f"{_MEM_PREFIX}peak":
                mem_for_phase[k[len(_MEM_PREFIX):]] = v

        # regular phase(exclude GPU Utilization indicators,total and mem@ indicator)
        for k in avg:
            if k != "total" and k not in _GPU_UTIL_KEYS and not k.startswith(_MEM_PREFIX):
                mem_str = f"  | mem {mem_for_phase[k]:.1f} GiB" if k in mem_for_phase else ""
                lines.append(f"  {k:30s} {avg[k]:8.1f} ms{mem_str}")
        if "total" in avg:
            peak_key = f"{_MEM_PREFIX}peak"
            peak_str = f"  | peak {avg[peak_key]:.1f} GiB" if peak_key in avg else ""
            lines.append(f"  {'total':30s} {avg['total']:8.1f} ms{peak_str}")

        # GPU utilization area
        if any(k in avg for k in _GPU_UTIL_KEYS):
            lines.append(f"  {'── GPU utilization ──':30s}")
            for k in ["guid_gpu_active", "overlap_guid_P3",
                       "train_gpu_idle", "guid_gpu_idle"]:
                if k in avg:
                    lines.append(f"  {k:30s} {avg[k]:8.1f} ms")

        logging.info("\n".join(lines))
