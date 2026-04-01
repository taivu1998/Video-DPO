import copy
import os
from typing import Any, Dict, Iterable, List, Optional

import yaml


DEFAULT_TRAINING_PROMPTS = [
    "cinematic shot, smooth motion, high quality, 8k",
    "drone footage flying over mountains, smooth camera, film quality",
    "slow motion video of water flowing, detailed, seamless",
    "timelapse of sunset clouds, smooth transition, vibrant colors",
    "walking through autumn forest, steady cam, golden hour lighting",
    "ocean waves crashing on beach, smooth motion, aerial view",
    "city street at night, smooth camera movement, neon lights",
    "wildlife documentary shot, smooth tracking, nature footage",
]

SUPPORTED_PRECISION_MODES = {"no", "fp16", "bf16"}
SUPPORTED_JITTER_METHODS = {"noise", "img2img"}
VALIDATION_PROFILES = {"common", "train", "generate", "inference", "evaluate"}


def _normalize_precision_mode(raw_mode: Any) -> str:
    if raw_mode is None or raw_mode is False:
        return "no"
    if raw_mode is True:
        return "fp16"
    return str(raw_mode).lower()


def _require_keys(section_name: str, section: Dict[str, Any], keys: Iterable[str]) -> None:
    for key in keys:
        if key not in section:
            raise ValueError(f"{section_name}.{key} is required")


def load_raw_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file from disk."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config at {config_path} must load to a dictionary")

    return loaded


def _normalize_prompt_values(raw_prompts: Any) -> List[str]:
    if raw_prompts is None:
        return []
    if isinstance(raw_prompts, str):
        prompts = [raw_prompts]
    elif isinstance(raw_prompts, Iterable):
        prompts = list(raw_prompts)
    else:
        raise ValueError("Prompts must be a string or a list of strings")

    normalized = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
    if any(not isinstance(prompt, str) for prompt in prompts):
        raise ValueError("All prompts must be strings")
    return normalized


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize configuration to a single internal shape."""
    normalized = copy.deepcopy(config)
    normalized.setdefault("data", {})
    normalized.setdefault("model", {})
    normalized.setdefault("training", {})
    normalized.setdefault("output_dir", "./checkpoints")
    normalized.setdefault("log_dir", "./logs")

    data = normalized["data"]
    training = normalized["training"]

    if "prompts" in data and data["prompts"] is not None:
        prompts = _normalize_prompt_values(data["prompts"])
    else:
        prompts = _normalize_prompt_values(data.get("prompt"))

    data["prompts"] = prompts
    if prompts:
        data["prompt"] = prompts[0]

    training["mixed_precision"] = _normalize_precision_mode(training.get("mixed_precision"))

    return normalized


def validate_config(config: Dict[str, Any], profile: str = "common") -> Dict[str, Any]:
    """Validate the normalized config and return it for convenient chaining."""
    if profile not in VALIDATION_PROFILES:
        raise ValueError(f"Unknown validation profile '{profile}'")

    required_sections = ("data", "model", "training")
    for section in required_sections:
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Config must include a '{section}' section")

    data = config["data"]
    model = config["model"]
    training = config["training"]

    if training["mixed_precision"] not in SUPPORTED_PRECISION_MODES:
        raise ValueError(
            f"Unsupported mixed_precision '{training['mixed_precision']}'. "
            f"Expected one of {sorted(SUPPORTED_PRECISION_MODES)}."
        )

    jitter_method = data.get("jitter_method")
    if jitter_method is not None and jitter_method not in SUPPORTED_JITTER_METHODS:
        raise ValueError(
            f"Unsupported jitter_method '{jitter_method}'. "
            f"Expected one of {sorted(SUPPORTED_JITTER_METHODS)}."
        )

    for field in ("batch_size", "gradient_accumulation_steps", "max_train_steps", "save_steps", "logging_steps"):
        if field in training and int(training[field]) <= 0:
            raise ValueError(f"training.{field} must be a positive integer")

    if "num_pairs" in data and int(data["num_pairs"]) <= 0:
        raise ValueError("data.num_pairs must be a positive integer")
    if "num_frames" in data and int(data["num_frames"]) <= 0:
        raise ValueError("data.num_frames must be a positive integer")
    if "resolution" in data and int(data["resolution"]) <= 0:
        raise ValueError("data.resolution must be a positive integer")

    if "base_model" not in model:
        raise ValueError("model.base_model is required")
    if "motion_adapter" not in model:
        raise ValueError("model.motion_adapter is required")

    if profile == "train":
        _require_keys("data", data, ("root_dir",))
        _require_keys("model", model, ("lora_rank", "lora_alpha"))
        _require_keys(
            "training",
            training,
            (
                "seed",
                "batch_size",
                "gradient_accumulation_steps",
                "learning_rate",
                "max_train_steps",
                "beta",
                "save_steps",
                "logging_steps",
            ),
        )

    if profile == "generate":
        _require_keys("data", data, ("root_dir", "num_pairs", "num_frames"))
        _require_keys("training", training, ("seed",))

    if profile == "evaluate":
        _require_keys("training", training, ("seed",))

    return config


def apply_overrides(
    config: Dict[str, Any],
    *,
    checkpoint: Optional[str] = None,
    seed: Optional[int] = None,
    num_frames: Optional[int] = None,
) -> Dict[str, Any]:
    updated = copy.deepcopy(config)
    updated.setdefault("training", {})
    updated.setdefault("data", {})

    if checkpoint is not None:
        updated["inference_checkpoint"] = checkpoint
    if seed is not None:
        updated["training"]["seed"] = seed
    if num_frames is not None:
        updated["data"]["num_frames"] = num_frames

    return updated


def load_config(
    config_path: str,
    *,
    checkpoint: Optional[str] = None,
    seed: Optional[int] = None,
    num_frames: Optional[int] = None,
    profile: str = "common",
) -> Dict[str, Any]:
    config = load_raw_config(config_path)
    config = apply_overrides(config, checkpoint=checkpoint, seed=seed, num_frames=num_frames)
    config = normalize_config(config)
    return validate_config(config, profile=profile)


def get_prompt_list(config: Dict[str, Any], fallback: Optional[Iterable[str]] = None) -> List[str]:
    data = config.get("data", {})
    prompts = _normalize_prompt_values(data.get("prompts"))
    if not prompts:
        prompts = _normalize_prompt_values(data.get("prompt"))
    if prompts:
        return prompts
    if fallback is None:
        raise ValueError("Config must define data.prompt or data.prompts")
    return _normalize_prompt_values(list(fallback))


def get_default_prompt(config: Dict[str, Any], fallback: Optional[Iterable[str]] = None) -> str:
    prompts = get_prompt_list(config, fallback=fallback)
    return prompts[0]


def resolve_prompt_override(
    config: Dict[str, Any],
    *,
    prompt: Optional[str] = None,
    prompt_index: Optional[int] = None,
    fallback: Optional[Iterable[str]] = None,
) -> str:
    if prompt is not None and prompt_index is not None:
        raise ValueError("Use either prompt or prompt_index, not both")

    prompts = get_prompt_list(config, fallback=fallback)
    if prompt is not None:
        stripped = prompt.strip()
        if not stripped:
            raise ValueError("Prompt override cannot be empty")
        return stripped

    if prompt_index is not None:
        if prompt_index < 0 or prompt_index >= len(prompts):
            raise IndexError(f"prompt_index {prompt_index} is out of range for {len(prompts)} prompt(s)")
        return prompts[prompt_index]

    return prompts[0]


def resolve_checkpoint_path(config: Dict[str, Any], checkpoint: Optional[str] = None) -> Optional[str]:
    if checkpoint is not None:
        return checkpoint
    return config.get("inference_checkpoint")
