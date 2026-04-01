import argparse
from typing import Any, Dict, List, Optional

from src.config import load_config as load_config_file


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    include_checkpoint: bool = False,
    include_seed: bool = True,
    include_num_frames: bool = False,
) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    if include_checkpoint:
        parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path override")
    if include_seed:
        parser.add_argument("--seed", type=int, default=None, help="Override seed")
    if include_num_frames:
        parser.add_argument("--num-frames", dest="num_frames", type=int, default=None, help="Override frame count")
    return parser


def build_common_parser(
    *,
    include_checkpoint: bool = False,
    include_seed: bool = True,
    include_num_frames: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    return add_common_args(
        parser,
        include_checkpoint=include_checkpoint,
        include_seed=include_seed,
        include_num_frames=include_num_frames,
    )


def load_config_from_namespace(args: Any, *, profile: str = "common") -> Dict[str, Any]:
    return load_config_file(
        args.config,
        checkpoint=getattr(args, "checkpoint", None),
        seed=getattr(args, "seed", None),
        num_frames=getattr(args, "num_frames", None),
        profile=profile,
    )


def load_config(
    argv: Optional[List[str]] = None,
    parser: Optional[argparse.ArgumentParser] = None,
    *,
    profile: str = "common",
) -> Dict[str, Any]:
    """
    Backward-compatible helper for scripts that only need config/checkpoint/seed.
    """
    active_parser = parser or build_common_parser(include_checkpoint=True, include_seed=True)
    args = active_parser.parse_args(argv)
    return load_config_from_namespace(args, profile=profile)
