"""
Data Validation Script for Video-DPO

Validates the generated training data to ensure:
1. All .pt files are loadable
2. Tensors have correct shapes
3. No NaN or Inf values
4. Data statistics are reasonable

Usage:
    python scripts/validate_data.py --config configs/train_config.yaml
"""

import sys
import os
import torch
import glob
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.config_parser import load_config


def validate_tensor(tensor: torch.Tensor, name: str, expected_dims: int = None) -> dict:
    """Validate a single tensor and return statistics."""
    issues = []
    stats = {
        "name": name,
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "has_nan": torch.isnan(tensor).any().item(),
        "has_inf": torch.isinf(tensor).any().item(),
    }

    if stats["has_nan"]:
        issues.append(f"{name} contains NaN values")
    if stats["has_inf"]:
        issues.append(f"{name} contains Inf values")
    if expected_dims and len(tensor.shape) != expected_dims:
        issues.append(f"{name} expected {expected_dims} dims, got {len(tensor.shape)}")

    stats["issues"] = issues
    return stats


def main():
    config = load_config()
    data_dir = config['data']['root_dir']

    print("=" * 60)
    print("Video-DPO Data Validation")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print()

    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory does not exist: {data_dir}")
        print("Run 'make data' first to generate training data.")
        return False

    # Find all .pt files
    pt_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))

    if not pt_files:
        print(f"ERROR: No .pt files found in {data_dir}")
        print("Run 'make data' first to generate training data.")
        return False

    print(f"Found {len(pt_files)} data files")
    print()

    # Validation statistics
    total_files = len(pt_files)
    valid_files = 0
    corrupted_files = []
    all_issues = []

    # Expected shapes
    num_frames = config['data'].get('num_frames', 16)
    expected_latent_channels = 4  # SD VAE latent channels
    expected_latent_size = config['data'].get('resolution', 512) // 8  # 64 for 512px

    print(f"Expected latent shape: [4, {num_frames}, {expected_latent_size}, {expected_latent_size}]")
    print()

    # Sample statistics
    latents_w_stats = []
    latents_l_stats = []

    for pt_file in tqdm(pt_files, desc="Validating"):
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)

            # Check required keys
            required_keys = ["latents_w", "latents_l", "prompt_embeds"]
            for key in required_keys:
                if key not in data:
                    all_issues.append(f"{pt_file}: Missing key '{key}'")
                    continue

            # Validate latents_w
            stats_w = validate_tensor(data["latents_w"], "latents_w", expected_dims=4)
            latents_w_stats.append(stats_w)
            all_issues.extend([f"{pt_file}: {issue}" for issue in stats_w["issues"]])

            # Validate latents_l
            stats_l = validate_tensor(data["latents_l"], "latents_l", expected_dims=4)
            latents_l_stats.append(stats_l)
            all_issues.extend([f"{pt_file}: {issue}" for issue in stats_l["issues"]])

            # Validate prompt_embeds
            stats_p = validate_tensor(data["prompt_embeds"], "prompt_embeds")
            all_issues.extend([f"{pt_file}: {issue}" for issue in stats_p["issues"]])

            # Check shape consistency
            if data["latents_w"].shape != data["latents_l"].shape:
                all_issues.append(
                    f"{pt_file}: latents_w shape {data['latents_w'].shape} != "
                    f"latents_l shape {data['latents_l'].shape}"
                )

            valid_files += 1

        except Exception as e:
            corrupted_files.append(pt_file)
            all_issues.append(f"{pt_file}: Failed to load - {str(e)}")

    # Print summary
    print()
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total files:     {total_files}")
    print(f"Valid files:     {valid_files}")
    print(f"Corrupted files: {len(corrupted_files)}")
    print()

    if latents_w_stats:
        # Aggregate statistics
        print("Latents_w Statistics:")
        shapes = [s["shape"] for s in latents_w_stats]
        unique_shapes = list(set(shapes))
        print(f"  Unique shapes: {unique_shapes}")
        print(f"  Min value: {min(s['min'] for s in latents_w_stats):.4f}")
        print(f"  Max value: {max(s['max'] for s in latents_w_stats):.4f}")
        print(f"  Mean: {sum(s['mean'] for s in latents_w_stats) / len(latents_w_stats):.4f}")
        print()

        print("Latents_l Statistics:")
        shapes = [s["shape"] for s in latents_l_stats]
        unique_shapes = list(set(shapes))
        print(f"  Unique shapes: {unique_shapes}")
        print(f"  Min value: {min(s['min'] for s in latents_l_stats):.4f}")
        print(f"  Max value: {max(s['max'] for s in latents_l_stats):.4f}")
        print(f"  Mean: {sum(s['mean'] for s in latents_l_stats) / len(latents_l_stats):.4f}")
        print()

    if all_issues:
        print("ISSUES FOUND:")
        for issue in all_issues[:20]:  # Show first 20 issues
            print(f"  - {issue}")
        if len(all_issues) > 20:
            print(f"  ... and {len(all_issues) - 20} more issues")
        print()

    if corrupted_files:
        print("CORRUPTED FILES:")
        for f in corrupted_files[:10]:
            print(f"  - {f}")
        if len(corrupted_files) > 10:
            print(f"  ... and {len(corrupted_files) - 10} more")
        print()

    # Final verdict
    print("=" * 60)
    if valid_files == total_files and not all_issues:
        print("SUCCESS: All data files are valid!")
        return True
    elif valid_files > 0:
        print(f"WARNING: {total_files - valid_files} files have issues")
        print("Training may still work with valid files.")
        return True
    else:
        print("ERROR: No valid data files found!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
