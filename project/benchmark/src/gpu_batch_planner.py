from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass
import torch


@dataclass
class BatchMeasurement:
    basins: int; starts: int; population: int; iterations: int; elapsed_seconds: float
    peak_allocated: int; peak_reserved: int; candidates_per_second: float; oom: bool = False; error: str = ""


def measure(adapter, *, basins: int, starts: int, population: int, iterations: int = 10) -> BatchMeasurement:
    device = adapter.device
    latent = torch.zeros((basins, starts, population, adapter.spec.dimension), device=device, dtype=torch.float64)
    if device.type != "cuda":
        t = time.perf_counter()
        for _ in range(iterations): adapter.evaluate(latent)
        return BatchMeasurement(basins, starts, population, iterations, time.perf_counter()-t, 0, 0, basins*starts*population*iterations/(time.perf_counter()-t))
    try:
        torch.cuda.reset_peak_memory_stats(device); adapter.evaluate(latent); torch.cuda.synchronize(device)
        t = time.perf_counter()
        for _ in range(iterations): adapter.evaluate(latent)
        torch.cuda.synchronize(device); elapsed = time.perf_counter()-t
        return BatchMeasurement(basins, starts, population, iterations, elapsed, torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device), basins*starts*population*iterations/elapsed)
    except torch.OutOfMemoryError as exc:
        return BatchMeasurement(basins, starts, population, iterations, 0, 0, 0, 0, True, repr(exc))
    finally:
        del latent; gc.collect()
