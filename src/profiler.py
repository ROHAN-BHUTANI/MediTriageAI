"""DGX Memory Profiling and Performance Benchmarking."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
import torch


@dataclass
class PerformanceMetrics:
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    peak_gpu_memory_mb: float
    cpu_ram_gb: float
    samples_per_sec: float
    tokens_per_sec: float
    steps_per_sec: float
    throughput_samples_per_sec: float
    wall_time_sec: float


class MemoryProfiler:
    def __init__(self, rank: int = 0):
        self.rank = rank
        self.start_time = None
        self.end_time = None
        self.total_samples = 0
        self.total_tokens = 0
        self.total_steps = 0

        # Memory baseline
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    def start(self):
        self.start_time = time.time()
        self.total_samples = 0
        self.total_tokens = 0
        self.total_steps = 0

    def step(self, batch_size: int, num_tokens: int):
        self.total_steps += 1
        self.total_samples += batch_size
        self.total_tokens += num_tokens

    def stop(self) -> PerformanceMetrics:
        self.end_time = time.time()
        wall_time = self.end_time - self.start_time

        # GPU Memory
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024**2)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**2)
            peak_gpu = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            gpu_allocated, gpu_reserved, peak_gpu = 0.0, 0.0, 0.0

        # CPU Memory
        cpu_ram = psutil.Process().memory_info().rss / (1024**3)

        # Throughput
        sps = self.total_samples / wall_time if wall_time > 0 else 0
        tps = self.total_tokens / wall_time if wall_time > 0 else 0
        steps_ps = self.total_steps / wall_time if wall_time > 0 else 0

        return PerformanceMetrics(
            gpu_allocated_mb=gpu_allocated,
            gpu_reserved_mb=gpu_reserved,
            peak_gpu_memory_mb=peak_gpu,
            cpu_ram_gb=cpu_ram,
            samples_per_sec=sps,
            tokens_per_sec=tps,
            steps_per_sec=steps_ps,
            throughput_samples_per_sec=sps,
            wall_time_sec=wall_time,
        )

    def export(self, metrics: PerformanceMetrics, output_path: Path):
        """Export metrics to JSON. Rank-safe."""
        if self.rank == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(asdict(metrics), f, indent=2)
