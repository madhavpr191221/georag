"""System and accelerator diagnostics with a real tensor execution check."""

from __future__ import annotations

import json
import platform
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


def select_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda:0")
    if mode == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raise ValueError("mode must be one of: auto, cpu, cuda")


def collect_diagnostics(mode: str = "auto", matrix_size: int = 256) -> dict[str, Any]:
    """Collect environment details and execute a finite matrix product."""

    if matrix_size <= 0:
        raise ValueError("matrix_size must be positive")
    device = select_device(mode)
    result = (
        torch.randn(matrix_size, matrix_size, device=device)
        @ torch.randn(matrix_size, matrix_size, device=device)
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    diagnostics: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "selected_device": str(device),
        "operation": "matrix_multiplication",
        "operation_shape": list(result.shape),
        "operation_finite": bool(torch.isfinite(result).all().item()),
    }
    if torch.cuda.is_available():
        diagnostics.update(
            cuda_device_count=torch.cuda.device_count(),
            cuda_device_name=torch.cuda.get_device_name(0),
            cuda_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        )
    return diagnostics


def write_diagnostics(path: str | Path, diagnostics: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    return destination
