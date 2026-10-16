"""Shared execution budgets for all DPPBench baselines."""
from __future__ import annotations

import contextlib
import contextvars
try:
    import resource
except ImportError:  # Windows: preserve actual process-memory accounting.
    resource = None
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator


class BudgetExceededError(RuntimeError):
    """Raised before an operation that would exceed the active budget."""


@dataclass
class Budget:
    max_pipeline_length: int | None = None
    max_evaluations: int | None = None
    max_memory_mb: float | None = None
    max_gpu_memory_mb: float | None = None
    max_llm_calls: int | None = None
    max_seconds: float | None = None
    max_output_rows: int | None = None
    max_output_columns: int | None = None

    evaluations_used: int = 0
    llm_calls_used: int = 0
    started_at: float = field(default_factory=time.monotonic)
    peak_memory_mb: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    _monitor_stop: threading.Event | None = field(default=None, init=False, repr=False)
    _monitor_thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "max_pipeline_length", "max_evaluations", "max_llm_calls",
            "max_output_rows", "max_output_columns",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("max_memory_mb", "max_gpu_memory_mb", "max_seconds"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def start(self) -> None:
        """Start timing the baseline phase without changing configured limits."""
        self.started_at = time.monotonic()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def _start_monitor(self) -> None:
        if self._monitor_thread is not None:
            return
        self._monitor_stop = threading.Event()

        def monitor() -> None:
            while self._monitor_stop is not None and not self._monitor_stop.wait(0.1):
                self.peak_memory_mb = max(self.peak_memory_mb, _process_peak_memory_mb())
                self.peak_gpu_memory_mb = max(self.peak_gpu_memory_mb, _gpu_peak_memory_mb())

        self._monitor_thread = threading.Thread(
            target=monitor, name="dppbench-budget-monitor", daemon=True,
        )
        self._monitor_thread.start()

    def _stop_monitor(self) -> None:
        if self._monitor_stop is not None:
            self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=1.0)
        self._monitor_stop = None
        self._monitor_thread = None

    def consume_evaluation(self) -> None:
        self.check_resources()
        if self.max_evaluations is not None and self.evaluations_used >= self.max_evaluations:
            raise BudgetExceededError(
                f"evaluation budget exhausted: {self.evaluations_used}/{self.max_evaluations}"
            )
        self.evaluations_used += 1

    def consume_llm_call(self) -> None:
        self.check_resources()
        if self.max_llm_calls is not None and self.llm_calls_used >= self.max_llm_calls:
            raise BudgetExceededError(
                f"LLM call budget exhausted: {self.llm_calls_used}/{self.max_llm_calls}"
            )
        self.llm_calls_used += 1

    def violations(self) -> list[str]:
        memory_mb = _process_peak_memory_mb()
        gpu_memory_mb = _gpu_peak_memory_mb()
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
        self.peak_gpu_memory_mb = max(self.peak_gpu_memory_mb, gpu_memory_mb)
        errors: list[str] = []
        if self.max_pipeline_length is not None and self.max_pipeline_length <= 0:
            errors.append("max_pipeline_length must be positive")
        if self.max_evaluations is not None and self.evaluations_used > self.max_evaluations:
            errors.append(
                f"evaluation budget exceeded: {self.evaluations_used}/{self.max_evaluations}"
            )
        if self.max_llm_calls is not None and self.llm_calls_used > self.max_llm_calls:
            errors.append(f"LLM call budget exceeded: {self.llm_calls_used}/{self.max_llm_calls}")
        if self.max_seconds is not None and self.elapsed_seconds > self.max_seconds:
            errors.append(
                f"wall-time budget exceeded: {self.elapsed_seconds:.1f}s/{self.max_seconds:.1f}s"
            )
        if self.max_memory_mb is not None and self.peak_memory_mb > self.max_memory_mb:
            errors.append(
                f"memory budget exceeded: {self.peak_memory_mb:.1f}MB/{self.max_memory_mb:.1f}MB"
            )
        if self.max_gpu_memory_mb is not None and self.peak_gpu_memory_mb > self.max_gpu_memory_mb:
            errors.append(
                f"GPU memory budget exceeded: {self.peak_gpu_memory_mb:.1f}MB/"
                f"{self.max_gpu_memory_mb:.1f}MB"
            )
        return errors

    def check_resources(self) -> None:
        errors = self.violations()
        if errors:
            raise BudgetExceededError("; ".join(errors))

    def snapshot(self) -> dict:
        self.violations()
        return {
            "max_pipeline_length": self.max_pipeline_length,
            "max_evaluations": self.max_evaluations,
            "max_memory_mb": self.max_memory_mb,
            "max_gpu_memory_mb": self.max_gpu_memory_mb,
            "max_llm_calls": self.max_llm_calls,
            "max_seconds": self.max_seconds,
            "max_output_rows": self.max_output_rows,
            "max_output_columns": self.max_output_columns,
            "evaluations_used": self.evaluations_used,
            "llm_calls_used": self.llm_calls_used,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "peak_memory_mb": round(self.peak_memory_mb, 3),
            "peak_gpu_memory_mb": round(self.peak_gpu_memory_mb, 3),
        }


def _process_peak_memory_mb() -> float:
    if resource is None:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                )
            ]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        # Budget tracks the peak within its own scope, not process lifetime.
        return counters.WorkingSetSize / (1024.0 * 1024.0)
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if value > 10_000_000 else value / 1024.0


def _gpu_peak_memory_mb() -> float:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        peak = max(
            int(torch.cuda.max_memory_allocated()),
            int(torch.cuda.max_memory_reserved()),
        )
        return float(peak) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


_ACTIVE_BUDGET: contextvars.ContextVar[Budget | None] = contextvars.ContextVar(
    "dppbench_active_budget", default=None
)


def get_active_budget() -> Budget | None:
    return _ACTIVE_BUDGET.get()


@contextlib.contextmanager
def budget_scope(budget: Budget | None) -> Iterator[Budget | None]:
    token = _ACTIVE_BUDGET.set(budget)
    if budget is not None:
        budget._start_monitor()
    try:
        yield budget
    finally:
        if budget is not None:
            budget._stop_monitor()
        _ACTIVE_BUDGET.reset(token)


__all__ = [
    "Budget", "BudgetExceededError", "budget_scope", "get_active_budget",
]
