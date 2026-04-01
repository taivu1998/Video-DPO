from typing import Any, Optional, Union

import torch


DeviceLike = Union[str, torch.device, None]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_device(device: DeviceLike = None) -> torch.device:
    if device is None:
        return get_device()
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def get_generator(seed: int, device: DeviceLike = None) -> torch.Generator:
    """Create a deterministic generator compatible with the selected device."""
    resolved_device = normalize_device(device)
    if resolved_device.type == "mps":
        return torch.Generator("cpu").manual_seed(seed)
    return torch.Generator(resolved_device).manual_seed(seed)


def _extract_precision_mode(config_or_mode: Any) -> str:
    if isinstance(config_or_mode, dict):
        return str(config_or_mode.get("training", {}).get("mixed_precision", "no")).lower()
    return str(config_or_mode or "no").lower()


def resolve_torch_dtype(config_or_mode: Any, device: DeviceLike = None) -> torch.dtype:
    """
    Resolve a safe torch dtype from a config precision mode and device.

    CUDA can honor fp16/bf16. MPS can honor fp16. CPU falls back to float32 for
    mixed-precision modes to avoid backend-specific surprises in model loading.
    """
    precision = _extract_precision_mode(config_or_mode)
    resolved_device = normalize_device(device)

    if precision == "fp16":
        if resolved_device.type in {"cuda", "mps"}:
            return torch.float16
        return torch.float32

    if precision == "bf16":
        if resolved_device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32

    return torch.float32
