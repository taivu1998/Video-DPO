import sys
import os
import torch
import argparse
from diffusers.utils import export_to_gif

sys.path.append(os.getcwd())
from src.config import resolve_checkpoint_path, resolve_prompt_override
from src.config_parser import add_common_args, load_config_from_namespace
from src.devices import get_device, get_generator
from src.model import VideoDPOModelWrapper

def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser, include_checkpoint=True, include_seed=True, include_num_frames=True)
    parser.add_argument("--prompt", type=str, default=None, help="Prompt override for generation")
    parser.add_argument("--prompt-index", type=int, default=None, help="Index into data.prompts")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for generated GIFs")
    args = parser.parse_args()

    config = load_config_from_namespace(args, profile="inference")
    device = get_device()
    checkpoint = resolve_checkpoint_path(config, args.checkpoint)

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
    prompt = resolve_prompt_override(config, prompt=args.prompt, prompt_index=args.prompt_index)
    seed = config['training'].get('seed', 42)

    print(f"Prompt: {prompt}")
    print(f"Seed: {seed}")

    # Create output directory
    output_dir = args.output_dir or config.get('output_dir', './checkpoints')
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
