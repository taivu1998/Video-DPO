import os
import tempfile
import unittest

import torch

from src.dataset import VideoDPODataset


class DatasetTests(unittest.TestCase):
    def _write_sample(self, directory: str, name: str, payload: dict) -> str:
        path = os.path.join(directory, name)
        torch.save(payload, path)
        return path

    def test_dataset_loads_and_squeezes_prompt_embeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_sample(
                tmpdir,
                "pair_00000.pt",
                {
                    "latents_w": torch.zeros(4, 8, 16, 16),
                    "latents_l": torch.ones(4, 8, 16, 16),
                    "prompt_embeds": torch.randn(1, 77, 768),
                },
            )
            dataset = VideoDPODataset(tmpdir)
            item = dataset[0]
            self.assertEqual(item["prompt_embeds"].dim(), 2)

    def test_dataset_rejects_mismatched_latent_shapes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_sample(
                tmpdir,
                "pair_00000.pt",
                {
                    "latents_w": torch.zeros(4, 8, 16, 16),
                    "latents_l": torch.ones(4, 7, 16, 16),
                    "prompt_embeds": torch.randn(1, 77, 768),
                },
            )
            dataset = VideoDPODataset(tmpdir)
            with self.assertRaises(ValueError):
                _ = dataset[0]


if __name__ == "__main__":
    unittest.main()
