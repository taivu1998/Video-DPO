import torch
from torch.utils.data import Dataset
import glob
import os
from typing import Dict


class VideoDPODataset(Dataset):
    """
    Loads pre-computed latent pairs.
    Structure: .pt file containing {'latents_w', 'latents_l', 'prompt_embeds'}

    Expected tensor shapes:
        - latents_w: [C, F, H, W] - Winner latents (temporally consistent)
        - latents_l: [C, F, H, W] - Loser latents (temporally jittery)
        - prompt_embeds: [1, seq_len, hidden_dim] - Text embeddings
    """
    def __init__(self, data_dir: str):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if not self.files:
            raise ValueError(f"No .pt files found in {data_dir}. Run 'make data' first.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load on CPU, let Accelerator move to GPU
        # Using weights_only=False as we're loading our own trusted data
        data = torch.load(self.files[idx], map_location="cpu", weights_only=False)

        required_keys = {"latents_w", "latents_l", "prompt_embeds"}
        missing_keys = required_keys.difference(data)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise KeyError(f"{self.files[idx]} is missing required keys: {missing}")

        # Ensure consistent dtype
        latents_w = data["latents_w"].float()
        latents_l = data["latents_l"].float()
        prompt_embeds = data["prompt_embeds"].float()

        if latents_w.dim() != 4 or latents_l.dim() != 4:
            raise ValueError(
                f"{self.files[idx]} must store 4D latent tensors, got "
                f"{tuple(latents_w.shape)} and {tuple(latents_l.shape)}"
            )
        if latents_w.shape != latents_l.shape:
            raise ValueError(
                f"{self.files[idx]} has mismatched latent shapes: "
                f"{tuple(latents_w.shape)} vs {tuple(latents_l.shape)}"
            )
        if prompt_embeds.dim() not in {2, 3}:
            raise ValueError(
                f"{self.files[idx]} must store prompt_embeds with 2 or 3 dims, "
                f"got shape {tuple(prompt_embeds.shape)}"
            )

        # Handle prompt_embeds shape - should be [seq_len, hidden_dim] for batching
        if prompt_embeds.dim() == 3 and prompt_embeds.shape[0] == 1:
            prompt_embeds = prompt_embeds.squeeze(0)
        elif prompt_embeds.dim() == 3:
            raise ValueError(
                f"{self.files[idx]} must store prompt_embeds with batch size 1 when 3D, "
                f"got shape {tuple(prompt_embeds.shape)}"
            )

        return {
            "latents_w": latents_w,
            "latents_l": latents_l,
            "prompt_embeds": prompt_embeds
        }
