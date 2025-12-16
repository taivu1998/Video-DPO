import sys
import os
import torch
import argparse
from diffusers.utils import export_to_gif

sys.path.append(os.getcwd())
from src.config_parser import load_config
from src.model import VideoDPOModelWrapper
from src.utils import get_device

def get_generator(seed: int, device: torch.device) -> torch.Generator:
    """Create a generator compatible with the device (MPS requires CPU generator)."""
    if device.type == "mps":
        return torch.Generator("cpu").manual_seed(seed)
    return torch.Generator(device).manual_seed(seed)

def main():
    config = load_config()
    device = get_device()
    checkpoint = config.get('inference_checkpoint')

    if not checkpoint:
        print("Error: Provide --checkpoint path")
        print("Usage: python scripts/inference.py --config configs/train_config.yaml --checkpoint checkpoints/latest")
        return

    if not os.path.exists(checkpoint):
        print(f"Error: Checkpoint not found at {checkpoint}")
        return

    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint}")

    wrapper = VideoDPOModelWrapper(config)
    num_frames = config['data'].get('num_frames', 16)
    prompt = config['data']['prompt']
    seed = config['training'].get('seed', 42)

    print(f"Prompt: {prompt}")
    print(f"Seed: {seed}")

    # Create output directory
    output_dir = config.get('output_dir', './checkpoints')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Base Model (Left Side)
    print("\n[1/2] Generating Base Model Reference...")
    pipe_base = wrapper.get_inference_pipeline(device)
    generator = get_generator(seed, device)
    base_out = pipe_base(
        prompt,
        num_frames=num_frames,
        num_inference_steps=25,
        guidance_scale=7.5,
        generator=generator
    ).frames[0]

    base_path = os.path.join(output_dir, "validation_base.gif")
    export_to_gif(base_out, base_path)
    print(f"Saved base model output to: {base_path}")

    # Free memory before loading DPO model
    del pipe_base
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 2. DPO Model (Right Side)
    print("\n[2/2] Generating DPO (Aligned) Result...")
    pipe_dpo = wrapper.get_inference_pipeline(device, lora_path=checkpoint)

    # Use SAME seed for direct comparison (Seed Lock)
    generator = get_generator(seed, device)
    dpo_out = pipe_dpo(
        prompt,
        num_frames=num_frames,
        num_inference_steps=25,
        guidance_scale=7.5,
        generator=generator
    ).frames[0]

    dpo_path = os.path.join(output_dir, "validation_dpo.gif")
    export_to_gif(dpo_out, dpo_path)
    print(f"Saved DPO model output to: {dpo_path}")

    print("\n" + "="*50)
    print("Validation Complete!")
    print(f"  Base Model:  {base_path}")
    print(f"  DPO Model:   {dpo_path}")
    print("\nCompare the two GIFs side-by-side to evaluate temporal consistency improvement.")
    print("="*50)

if __name__ == "__main__":
    main()