"""System and resource metrics for one service execution."""

from __future__ import annotations

import csv
import io
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List

try:
    import psutil
except Exception:  # pragma: no cover - exercised when optional runtime deps are absent
    psutil = None

MB = 1024 * 1024


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        f = float(value)
        if f != f:
            return None
        return f
    except Exception:
        return None


def _query_nvidia_gpu_snapshot() -> list[dict[str, Any]]:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return []
    try:
        result = subprocess.run(
            [
                binary,
                "--query-gpu=name,memory.total,memory.used,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    gpus: list[dict[str, Any]] = []
    for raw_line in result.stdout.splitlines():
        parts = [part.strip() for part in raw_line.strip().split(",")]
        if len(parts) != 5:
            continue
        name, mem_total, mem_used, util_gpu, util_mem = parts
        gpus.append(
            {
                "name": name,
                "memory_total_mb": _safe_float(mem_total),
                "memory_used_mb": _safe_float(mem_used),
                "utilization": _safe_float(util_gpu),
                "memory_utilization": _safe_float(util_mem),
            }
        )
    return gpus


def _extract_luid(instance_name: str) -> str | None:
    match = re.search(r"luid_([^_]+_[^_]+)", instance_name)
    return match.group(1) if match else None


def _query_windows_gpu_snapshot() -> list[dict[str, Any]]:
    if platform.system().lower() != "windows":
        return []
    typeperf = shutil.which("typeperf")
    if not typeperf:
        return []

    counters = [
        r"\GPU Engine(*)\Utilization Percentage",
        r"\GPU Adapter Memory(*)\Dedicated Usage",
        r"\GPU Adapter Memory(*)\Shared Usage",
    ]
    try:
        result = subprocess.run([typeperf, *counters, "-sc", "1"], check=True, capture_output=True, text=True)
    except Exception:
        return []

    rows = list(csv.reader(io.StringIO(result.stdout)))
    if len(rows) < 3:
        return []
    headers = rows[0]
    values = rows[-1]
    if len(headers) != len(values):
        return []

    adapters: dict[str, dict[str, Any]] = {}
    for header, raw_value in zip(headers[1:], values[1:]):
        value = _safe_float(raw_value.strip().strip('"'))
        if value is None:
            continue
        counter_name = header.strip().strip('"')
        luid = _extract_luid(counter_name) or "global"
        item = adapters.setdefault(
            luid,
            {
                "name": f"Windows GPU ({luid})",
                "memory_total_mb": None,
                "memory_used_mb": 0.0,
                "utilization": 0.0,
                "memory_utilization": None,
            },
        )
        lower = counter_name.lower()
        if "\\gpu engine(" in lower and "utilization percentage" in lower:
            item["utilization"] = max(_safe_float(item.get("utilization")) or 0.0, value)
        elif "\\gpu adapter memory(" in lower and ("dedicated usage" in lower or "shared usage" in lower):
            item["memory_used_mb"] = (_safe_float(item.get("memory_used_mb")) or 0.0) + (value / MB)
    return list(adapters.values())


def query_gpu_snapshot() -> list[dict[str, Any]]:
    return _query_nvidia_gpu_snapshot() or _query_windows_gpu_snapshot()


def capture_hardware_snapshot() -> dict[str, Any]:
    try:
        if psutil is None:
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "processor": platform.processor(),
                "machine": platform.machine(),
                "cpu": None,
                "memory": None,
                "gpu": [],
                "environment": {"pid": os.getpid(), "working_directory": os.getcwd()},
            }
        freq = psutil.cpu_freq()
        cpu_freq = {
            "current_mhz": _safe_float(freq.current) if freq else None,
            "min_mhz": _safe_float(freq.min) if freq else None,
            "max_mhz": _safe_float(freq.max) if freq else None,
        }
    except Exception:
        cpu_freq = None

    try:
        vm = psutil.virtual_memory()
        memory = {"total_mb": _safe_float(vm.total / MB), "available_mb": _safe_float(vm.available / MB)}
    except Exception:
        memory = None

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "frequency": cpu_freq,
        },
        "memory": memory,
        "gpu": query_gpu_snapshot(),
        "environment": {"pid": os.getpid(), "working_directory": os.getcwd()},
    }


@dataclass
class ResourceUsage:
    cpu_time_s: float | None
    cpu_utilization: float | None
    memory_used_mb: float | None
    memory_utilization: float | None
    gpu_utilization: float | None
    gpu_memory_utilization: float | None
    gpu_memory_used_mb: float | None
    peak_vram_mb: float | None
    avg_vram_mb: float | None
    peak_host_ram_mb: float | None
    avg_host_ram_mb: float | None


class ResourceTracker:
    def __init__(self) -> None:
        if psutil is None:
            self._process = None
            self._cpu_times_start = None
            self._cpu_count = 1
            self._sample_interval_s = 0.2
            self._samples = []
            self._lock = threading.Lock()
            self._running = False
            self._sampler_thread = None
            return
        self._process = psutil.Process(os.getpid())
        self._cpu_times_start = None
        self._cpu_count = psutil.cpu_count(logical=True) or 1
        self._sample_interval_s = 0.2
        self._samples: list[dict[str, float | None]] = []
        self._lock = threading.Lock()
        self._running = False
        self._sampler_thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        gpu_snapshot = query_gpu_snapshot()
        gpu_mem_used = _aggregate_gpu_sum(gpu_snapshot, "memory_used_mb")
        try:
            if psutil is None:
                host_used_mb = None
            else:
                host_used_mb = psutil.virtual_memory().used / MB
        except Exception:
            host_used_mb = None
        with self._lock:
            self._samples.append(
                {
                    "gpu_memory_used_mb": _safe_float(gpu_mem_used),
                    "host_ram_used_mb": _safe_float(host_used_mb),
                }
            )

    def _run_sampler(self) -> None:
        while self._running:
            self._sample_once()
            time.sleep(self._sample_interval_s)

    def start(self) -> None:
        if psutil is None or self._process is None:
            with self._lock:
                self._samples = []
            self._running = False
            return
        try:
            self._process.cpu_percent(None)
            self._cpu_times_start = self._process.cpu_times()
        except Exception:
            self._cpu_times_start = None
        with self._lock:
            self._samples = []
        self._running = True
        self._sample_once()
        self._sampler_thread = threading.Thread(target=self._run_sampler, daemon=True)
        self._sampler_thread.start()

    def stop(self, duration_s: float | None = None) -> ResourceUsage:
        if psutil is None or self._process is None:
            return ResourceUsage(
                cpu_time_s=None,
                cpu_utilization=None,
                memory_used_mb=None,
                memory_utilization=None,
                gpu_utilization=None,
                gpu_memory_utilization=None,
                gpu_memory_used_mb=None,
                peak_vram_mb=None,
                avg_vram_mb=None,
                peak_host_ram_mb=None,
                avg_host_ram_mb=None,
            )
        self._running = False
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=1.0)
        self._sample_once()
        cpu_time_s = None
        try:
            end_times = self._process.cpu_times()
            if self._cpu_times_start is not None:
                cpu_time_s = (end_times.user - self._cpu_times_start.user) + (end_times.system - self._cpu_times_start.system)
        except Exception:
            cpu_time_s = None

        cpu_util = None
        if cpu_time_s is not None and duration_s and duration_s > 0:
            cpu_util = max(0.0, min(100.0, (cpu_time_s / duration_s) / self._cpu_count * 100.0))

        try:
            memory_used_mb = self._process.memory_info().rss / MB
        except Exception:
            memory_used_mb = None
        try:
            memory_util = psutil.virtual_memory().percent
        except Exception:
            memory_util = None

        gpu_snapshot = query_gpu_snapshot()
        gpu_util = _aggregate_gpu_metric(gpu_snapshot, "utilization")
        gpu_mem_util = _aggregate_gpu_metric(gpu_snapshot, "memory_utilization")
        gpu_mem_used = _aggregate_gpu_sum(gpu_snapshot, "memory_used_mb")
        with self._lock:
            samples = list(self._samples)

        vram_samples = [s.get("gpu_memory_used_mb") for s in samples if s.get("gpu_memory_used_mb") is not None]
        host_ram_samples = [s.get("host_ram_used_mb") for s in samples if s.get("host_ram_used_mb") is not None]

        return ResourceUsage(
            cpu_time_s=_safe_float(cpu_time_s),
            cpu_utilization=_safe_float(cpu_util),
            memory_used_mb=_safe_float(memory_used_mb),
            memory_utilization=_safe_float(memory_util),
            gpu_utilization=_safe_float(gpu_util),
            gpu_memory_utilization=_safe_float(gpu_mem_util),
            gpu_memory_used_mb=_safe_float(gpu_mem_used),
            peak_vram_mb=_safe_float(max(vram_samples) if vram_samples else None),
            avg_vram_mb=_safe_float(mean(vram_samples) if vram_samples else None),
            peak_host_ram_mb=_safe_float(max(host_ram_samples) if host_ram_samples else None),
            avg_host_ram_mb=_safe_float(mean(host_ram_samples) if host_ram_samples else None),
        )


def _aggregate_gpu_metric(gpus: Iterable[Dict[str, Any]], key: str) -> float | None:
    values = [_safe_float(g.get(key)) for g in gpus]
    values = [v for v in values if v is not None]
    return max(values) if values else None


def _aggregate_gpu_sum(gpus: Iterable[Dict[str, Any]], key: str) -> float | None:
    values = [_safe_float(g.get(key)) for g in gpus]
    values = [v for v in values if v is not None]
    return sum(values) if values else None
