import sys
import os
import torch
import numpy as np
import gc
from diffusers import AnimateDiffPipeline, MotionAdapter, StableDiffusionImg2ImgPipeline
from diffusers.schedulers import DDIMScheduler
from tqdm import tqdm
from PIL import Image

sys.path.append(os.getcwd())
from src.config_parser import load_config
from src.utils import seed_everything, get_device

# Default diverse prompts for better generalization
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

def get_generator(seed: int, device: torch.device) -> torch.Generator:
    """Create a generator compatible with the device (MPS requires CPU generator)."""
    if device.type == "mps":
        return torch.Generator("cpu").manual_seed(seed)
    return torch.Generator(device).manual_seed(seed)

def add_temporal_jitter(frames: torch.Tensor, jitter_strength: float = 0.15) -> torch.Tensor:
    """
    Add temporal jitter to frames using direct noise injection.
    This is a faster alternative to img2img that creates temporal inconsistency.

    Args:
        frames: Tensor of shape [F, C, H, W] with values in [0, 1]
        jitter_strength: Amount of noise/color shift to add per frame

    Returns:
        Jittered frames tensor
    """
    jittered = frames.clone()
    num_frames = frames.shape[0]

    for i in range(num_frames):
        # Per-frame random noise
        frame_noise = torch.randn_like(jittered[i]) * jitter_strength
        # Slight color shift per frame for more obvious temporal inconsistency
        color_shift = (torch.rand(3, 1, 1, device=frames.device) - 0.5) * jitter_strength * 0.5
        jittered[i] = torch.clamp(jittered[i] + frame_noise + color_shift, 0, 1)

    return jittered

def main():
    config = load_config()
    device = get_device()
    seed_everything(config['training']['seed'])

    print(f"Using device: {device}")

    # Get data generation parameters
    data_config = config['data']
    num_pairs = data_config['num_pairs']
    num_frames = data_config['num_frames']
    resolution = data_config.get('resolution', 512)
    jitter_strength = data_config.get('jitter_strength', 0.15)
    jitter_method = data_config.get('jitter_method', 'noise')  # 'noise' or 'img2img'

    # Get prompts - support multiple prompts for diversity
    prompts = data_config.get('prompts', None)
    if prompts is None:
        # Fall back to single prompt or default prompts
        single_prompt = data_config.get('prompt', None)
        if single_prompt:
            prompts = [single_prompt]
        else:
            prompts = DEFAULT_TRAINING_PROMPTS
            print("Using default diverse training prompts")

    if isinstance(prompts, str):
        prompts = [prompts]

    print(f"\nData Generation Configuration:")
    print(f"  - Pairs: {num_pairs}")
    print(f"  - Frames: {num_frames}")
    print(f"  - Resolution: {resolution}x{resolution}")
    print(f"  - Jitter method: {jitter_method}")
    print(f"  - Jitter strength: {jitter_strength}")
    print(f"  - Prompts: {len(prompts)} diverse prompts")

    # 1. Load AnimateDiff (Winner Generator)
    print("\nLoading AnimateDiff Pipeline...")
    adapter = MotionAdapter.from_pretrained(
        config['model']['motion_adapter'],
        torch_dtype=torch.float16
    )
    pipe_winner = AnimateDiffPipeline.from_pretrained(
        config['model']['base_model'],
        motion_adapter=adapter,
        torch_dtype=torch.float16
    ).to(device)

    # Use DDIM scheduler for faster generation
    pipe_winner.scheduler = DDIMScheduler.from_config(
        pipe_winner.scheduler.config,
        beta_schedule="linear",
        clip_sample=False
    )
    pipe_winner.enable_vae_slicing()

    # 2. Load Img2Img (Loser Generator) only if using img2img method
    pipe_loser = None
    if jitter_method == 'img2img':
        print("Loading Img2Img Pipeline for jitter generation...")
        pipe_loser = StableDiffusionImg2ImgPipeline.from_pretrained(
            config['model']['base_model'],
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(device)

    os.makedirs(data_config['root_dir'], exist_ok=True)

    # Get VAE for encoding
    vae = pipe_winner.vae

    def encode_frames(pil_frames: list) -> torch.Tensor:
        """Encode PIL frames to latent space."""
        # Transform to tensor [F, C, H, W]
        tensors = torch.stack([
            torch.from_numpy(np.array(f.convert("RGB"))).permute(2, 0, 1).float() / 127.5 - 1.0
            for f in pil_frames
        ]).to(device, dtype=torch.float16)

        with torch.no_grad():
            latents = vae.encode(tensors).latent_dist.sample() * vae.config.scaling_factor

        # Reshape to [C, F, H, W] for AnimateDiff training format
        return latents.permute(1, 0, 2, 3).cpu().float()

    def encode_tensor_frames(tensor_frames: torch.Tensor) -> torch.Tensor:
        """Encode tensor frames [F, C, H, W] in [0, 1] range to latent space."""
        # Convert from [0, 1] to [-1, 1]
        tensors = (tensor_frames * 2 - 1).to(device, dtype=torch.float16)

        with torch.no_grad():
            latents = vae.encode(tensors).latent_dist.sample() * vae.config.scaling_factor

        return latents.permute(1, 0, 2, 3).cpu().float()

    print(f"\nGenerating {num_pairs} pairs...")

    for i in tqdm(range(num_pairs), desc="Generating pairs"):
        try:
            # Cycle through prompts for diversity
            prompt = prompts[i % len(prompts)]

            # Pre-encode prompt for this pair
            with torch.no_grad():
                text_inputs = pipe_winner.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=pipe_winner.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                prompt_embeds = pipe_winner.text_encoder(text_inputs.input_ids.to(device))[0].cpu()

            # --- A. Generate Winner (Temporally Consistent) ---
            generator = get_generator(config['training']['seed'] + i, device)
            result = pipe_winner(
                prompt,
                num_frames=num_frames,
                height=resolution,
                width=resolution,
                num_inference_steps=25,
                guidance_scale=7.5,
                generator=generator,
                output_type="pt" if jitter_method == 'noise' else "pil"
            )

            if jitter_method == 'noise':
                # Output is tensor [1, F, C, H, W] or [F, C, H, W]
                frames_w_tensor = result.frames[0]  # [F, C, H, W]
                if frames_w_tensor.dim() == 5:
                    frames_w_tensor = frames_w_tensor.squeeze(0)

                # --- B. Generate Loser (Direct Noise Jitter) ---
                frames_l_tensor = add_temporal_jitter(frames_w_tensor, jitter_strength)

                # --- C. Encode to Latents ---
                latents_w = encode_tensor_frames(frames_w_tensor)
                latents_l = encode_tensor_frames(frames_l_tensor)
            else:
                # PIL output for img2img method
                frames_w = result.frames[0]  # List of PIL images

                # --- B. Generate Loser (Img2Img Jitter) ---
                frames_l = []
                for frame_idx, frame in enumerate(frames_w):
                    unique_seed = i * 1000 + frame_idx + torch.randint(0, 10000, (1,)).item()
                    gen_l = get_generator(unique_seed, device)

                    jittered = pipe_loser(
                        prompt=prompt,
                        image=frame,
                        strength=jitter_strength,
                        guidance_scale=7.5,
                        num_inference_steps=20,
                        generator=gen_l
                    ).images[0]
                    frames_l.append(jittered)

                # --- C. Encode to Latents ---
                latents_w = encode_frames(frames_w)
                latents_l = encode_frames(frames_l)

            # Save Pair
            torch.save({
                "latents_w": latents_w,  # Shape: [C, F, H, W]
                "latents_l": latents_l,  # Shape: [C, F, H, W]
                "prompt_embeds": prompt_embeds  # Shape: [1, seq_len, hidden_dim]
            }, os.path.join(data_config['root_dir'], f"pair_{i:05d}.pt"))

            # Save example GIFs for first pair
            if i == 0:
                from diffusers.utils import export_to_gif
                if jitter_method == 'noise':
                    # Convert tensors to PIL for GIF export
                    winner_pil = [Image.fromarray((f.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                                  for f in frames_w_tensor]
                    loser_pil = [Image.fromarray((f.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                                 for f in frames_l_tensor]
                else:
                    winner_pil = frames_w
                    loser_pil = frames_l

                export_to_gif(winner_pil, os.path.join(data_config['root_dir'], "example_winner.gif"))
                export_to_gif(loser_pil, os.path.join(data_config['root_dir'], "example_loser.gif"))
                print(f"\nSaved example GIFs (Prompt: {prompt})")

            # Free memory periodically
            if (i + 1) % 5 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()

            if (i + 1) % 50 == 0:
                print(f"Completed {i + 1}/{num_pairs} pairs")

        except Exception as e:
            print(f"Error generating pair {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Cleanup
    del pipe_winner
    if pipe_loser:
        del pipe_loser
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    print(f"\nData generation complete! Generated {num_pairs} pairs in {data_config['root_dir']}")

if __name__ == "__main__":
    main()