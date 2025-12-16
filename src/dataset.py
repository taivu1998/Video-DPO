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

        # Ensure consistent dtype
        latents_w = data["latents_w"].float()
        latents_l = data["latents_l"].float()
        prompt_embeds = data["prompt_embeds"].float()

        # Handle prompt_embeds shape - should be [seq_len, hidden_dim] for batching
        if prompt_embeds.dim() == 3 and prompt_embeds.shape[0] == 1:
            prompt_embeds = prompt_embeds.squeeze(0)

        return {
            "latents_w": latents_w,
            "latents_l": latents_l,
            "prompt_embeds": prompt_embeds
        }