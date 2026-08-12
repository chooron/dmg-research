from __future__ import annotations

import time
from dataclasses import dataclass
import torch


@dataclass
class CompileResult:
    mode: str
    function: object
    compile_seconds: float
    error: str | None = None


def compile_kernel(fn, sample: torch.Tensor, mode: str) -> CompileResult:
    """Compile a numeric closure only; Python orchestration remains eager."""
    if mode == "eager": return CompileResult(mode, fn, 0.0)
    started = time.perf_counter()
    try:
        compiled = torch.compile(fn, backend="inductor", mode=mode, fullgraph=False)
        with torch.inference_mode(): compiled(sample)
        if sample.is_cuda: torch.cuda.synchronize(sample.device)
        return CompileResult(mode, compiled, time.perf_counter() - started)
    except Exception as exc:
        return CompileResult(mode, fn, time.perf_counter() - started, f"{type(exc).__name__}: {exc}")
