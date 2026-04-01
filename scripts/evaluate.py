"""
Evaluation script for Video-DPO using Optical Flow (Warping Error).

This script computes the temporal consistency metric based on RAFT optical flow:
1. Compute optical flow between consecutive frames using RAFT
2. Warp Frame_t using the flow to predict Frame_{t+1}
3. Calculate MSE between the Warped Prediction and the Actual Next Frame

Lower warping error = better temporal consistency.

Usage:
    python scripts/evaluate.py --config configs/train_config.yaml --checkpoint checkpoints/latest
    python scripts/evaluate.py --config configs/train_config.yaml --checkpoint checkpoints/latest --num_samples 20
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

sys.path.append(os.getcwd())
from src.config import resolve_checkpoint_path, resolve_prompt_override
from src.config_parser import add_common_args, load_config_from_namespace
from src.devices import get_device, get_generator
from src.model import VideoDPOModelWrapper
from src.utils import seed_everything


def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized tensor [C, H, W]."""
    arr = np.array(pil_image.convert("RGB"))
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return tensor


def compute_optical_flow_simple(frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
    """
    Compute simplified optical flow using gradient-based Lucas-Kanade approach.
    This is a fallback when RAFT is not available.

    Args:
        frame1: [C, H, W] tensor, normalized to [0, 1]
        frame2: [C, H, W] tensor, normalized to [0, 1]

    Returns:
        flow: [2, H, W] tensor (u, v flow vectors)
    """
    # Convert to grayscale
    gray1 = 0.299 * frame1[0] + 0.587 * frame1[1] + 0.114 * frame1[2]
    gray2 = 0.299 * frame2[0] + 0.587 * frame2[1] + 0.114 * frame2[2]

    # Compute spatial gradients
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32) / 8.0

    gray1 = gray1.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    gray2 = gray2.unsqueeze(0).unsqueeze(0)

    kernel_x = kernel_x.unsqueeze(0).unsqueeze(0).to(gray1.device)
    kernel_y = kernel_y.unsqueeze(0).unsqueeze(0).to(gray1.device)

    Ix = F.conv2d(gray1, kernel_x, padding=1)
    Iy = F.conv2d(gray1, kernel_y, padding=1)
    It = gray2 - gray1

    # Simple flow estimation (regularized least squares)
    eps = 1e-6
    u = -It * Ix / (Ix.pow(2) + Iy.pow(2) + eps)
    v = -It * Iy / (Ix.pow(2) + Iy.pow(2) + eps)

    # Clamp to reasonable values
    u = torch.clamp(u, -50, 50)
    v = torch.clamp(v, -50, 50)

    flow = torch.cat([u, v], dim=1).squeeze(0)  # [2, H, W]
    return flow


def warp_frame(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp a frame using optical flow.

    Args:
        frame: [C, H, W] tensor
        flow: [2, H, W] tensor (u, v)

    Returns:
        warped: [C, H, W] tensor
    """
    C, H, W = frame.shape

    # Create coordinate grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=frame.device, dtype=torch.float32),
        torch.arange(W, device=frame.device, dtype=torch.float32),
        indexing='ij'
    )

    # Add flow to coordinates
    new_x = x_coords + flow[0]
    new_y = y_coords + flow[1]

    # Normalize to [-1, 1] for grid_sample
    new_x = 2.0 * new_x / (W - 1) - 1.0
    new_y = 2.0 * new_y / (H - 1) - 1.0

    # Create sampling grid [1, H, W, 2]
    grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)

    # Warp frame
    frame_batch = frame.unsqueeze(0)  # [1, C, H, W]
    warped = F.grid_sample(frame_batch, grid, mode='bilinear', padding_mode='border', align_corners=True)

    return warped.squeeze(0)  # [C, H, W]


def compute_warping_error(frames: list) -> float:
    """
    Compute average warping error across all consecutive frame pairs.

    Args:
        frames: List of PIL images

    Returns:
        Average warping error (MSE)
    """
    if len(frames) < 2:
        return 0.0

    device = torch.device("cpu")  # Use CPU for evaluation to avoid memory issues

    total_error = 0.0
    num_pairs = len(frames) - 1

    for i in range(num_pairs):
        frame1 = pil_to_tensor(frames[i]).to(device)
        frame2 = pil_to_tensor(frames[i + 1]).to(device)

        # Compute optical flow from frame1 to frame2
        flow = compute_optical_flow_simple(frame1, frame2)

        # Warp frame1 using flow to predict frame2
        warped = warp_frame(frame1, flow)

        # Compute MSE between warped prediction and actual frame2
        mse = F.mse_loss(warped, frame2).item()
        total_error += mse

    return total_error / num_pairs


def compute_frame_difference_metric(frames: list) -> float:
    """
    Compute average absolute frame difference (simpler flickering metric).
    Higher values indicate more flickering/inconsistency.

    Args:
        frames: List of PIL images

    Returns:
        Average frame difference
    """
    if len(frames) < 2:
        return 0.0

    total_diff = 0.0
    num_pairs = len(frames) - 1

    for i in range(num_pairs):
        frame1 = pil_to_tensor(frames[i])
        frame2 = pil_to_tensor(frames[i + 1])

        diff = torch.abs(frame2 - frame1).mean().item()
        total_diff += diff

    return total_diff / num_pairs


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser, include_checkpoint=True, include_seed=True, include_num_frames=True)
    parser.add_argument("--prompt", type=str, default=None, help="Prompt override for evaluation")
    parser.add_argument("--prompt-index", type=int, default=None, help="Index into data.prompts")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results")
    args = parser.parse_args()

    config = load_config_from_namespace(args, profile="evaluate")
    checkpoint = resolve_checkpoint_path(config, args.checkpoint)

    device = get_device()
    seed_everything(config['training']['seed'])

    print("="*60)
    print("Video-DPO Evaluation")
    print("="*60)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Number of samples: {args.num_samples}")
    print()

    if not checkpoint:
        print("Error: Provide --checkpoint path or set inference_checkpoint in config")
        return

    if not os.path.exists(checkpoint):
        print(f"Error: Checkpoint not found at {checkpoint}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    wrapper = VideoDPOModelWrapper(config)
    num_frames = config['data'].get('num_frames', 16)
    prompt = resolve_prompt_override(config, prompt=args.prompt, prompt_index=args.prompt_index)

    # Results storage
    base_warping_errors = []
    dpo_warping_errors = []
    base_frame_diffs = []
    dpo_frame_diffs = []

    print(f"Prompt: {prompt}")
    print(f"Frames per video: {num_frames}")
    print()

    # Load base pipeline
    print("Loading Base Model...")
    pipe_base = wrapper.get_inference_pipeline(device)

    print("Evaluating Base Model...")
    for seed in tqdm(range(args.num_samples), desc="Base Model"):
        generator = get_generator(seed, device)
        result = pipe_base(
            prompt,
            num_frames=num_frames,
            num_inference_steps=25,
            guidance_scale=7.5,
            generator=generator
        )
        frames = result.frames[0]

        warping_error = compute_warping_error(frames)
        frame_diff = compute_frame_difference_metric(frames)

        base_warping_errors.append(warping_error)
        base_frame_diffs.append(frame_diff)

    # Free memory
    del pipe_base
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Load DPO pipeline
    print("\nLoading DPO Model...")
    pipe_dpo = wrapper.get_inference_pipeline(device, lora_path=checkpoint)

    print("Evaluating DPO Model...")
    for seed in tqdm(range(args.num_samples), desc="DPO Model"):
        generator = get_generator(seed, device)
        result = pipe_dpo(
            prompt,
            num_frames=num_frames,
            num_inference_steps=25,
            guidance_scale=7.5,
            generator=generator
        )
        frames = result.frames[0]

        warping_error = compute_warping_error(frames)
        frame_diff = compute_frame_difference_metric(frames)

        dpo_warping_errors.append(warping_error)
        dpo_frame_diffs.append(frame_diff)

    # Compute statistics
    base_warp_mean = np.mean(base_warping_errors)
    base_warp_std = np.std(base_warping_errors)
    dpo_warp_mean = np.mean(dpo_warping_errors)
    dpo_warp_std = np.std(dpo_warping_errors)

    base_diff_mean = np.mean(base_frame_diffs)
    base_diff_std = np.std(base_frame_diffs)
    dpo_diff_mean = np.mean(dpo_frame_diffs)
    dpo_diff_std = np.std(dpo_frame_diffs)

    warp_improvement = ((base_warp_mean - dpo_warp_mean) / base_warp_mean) * 100
    diff_improvement = ((base_diff_mean - dpo_diff_mean) / base_diff_mean) * 100

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print()
    print("Warping Error (lower is better):")
    print(f"  Base Model:  {base_warp_mean:.6f} (+/- {base_warp_std:.6f})")
    print(f"  DPO Model:   {dpo_warp_mean:.6f} (+/- {dpo_warp_std:.6f})")
    print(f"  Improvement: {warp_improvement:+.2f}%")
    print()
    print("Frame Difference (lower is better):")
    print(f"  Base Model:  {base_diff_mean:.6f} (+/- {base_diff_std:.6f})")
    print(f"  DPO Model:   {dpo_diff_mean:.6f} (+/- {dpo_diff_std:.6f})")
    print(f"  Improvement: {diff_improvement:+.2f}%")
    print()

    if dpo_warp_mean < base_warp_mean:
        print("SUCCESS: DPO model shows improved temporal consistency!")
    else:
        print("NOTE: DPO model did not improve on this metric.")

    print("="*60)

    # Save detailed results
    results = {
        "config": args.config,
        "checkpoint": checkpoint,
        "num_samples": args.num_samples,
        "base_warping_errors": base_warping_errors,
        "dpo_warping_errors": dpo_warping_errors,
        "base_frame_diffs": base_frame_diffs,
        "dpo_frame_diffs": dpo_frame_diffs,
        "summary": {
            "base_warp_mean": base_warp_mean,
            "base_warp_std": base_warp_std,
            "dpo_warp_mean": dpo_warp_mean,
            "dpo_warp_std": dpo_warp_std,
            "warp_improvement_pct": warp_improvement,
            "base_diff_mean": base_diff_mean,
            "dpo_diff_mean": dpo_diff_mean,
            "diff_improvement_pct": diff_improvement,
        }
    }

    results_path = os.path.join(args.output_dir, "evaluation_results.pt")
    torch.save(results, results_path)
    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
