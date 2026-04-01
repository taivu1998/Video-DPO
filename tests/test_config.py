import argparse
import tempfile
import textwrap
import unittest

from src.config import (
    get_default_prompt,
    get_prompt_list,
    load_config,
    resolve_checkpoint_path,
    resolve_prompt_override,
)
from src.config_parser import load_config_from_namespace


class ConfigTests(unittest.TestCase):
    def _write_config(self, body: str) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        handle.write(textwrap.dedent(body))
        handle.flush()
        handle.close()
        self.addCleanup(lambda: __import__("os").unlink(handle.name))
        return handle.name

    def test_load_config_normalizes_prompts_list(self):
        path = self._write_config(
            """
            output_dir: ./checkpoints
            log_dir: ./logs
            data:
              prompt: "one prompt"
              root_dir: ./data
              num_pairs: 1
              num_frames: 8
              resolution: 256
            model:
              base_model: base
              motion_adapter: adapter
            training:
              seed: 42
              batch_size: 1
              gradient_accumulation_steps: 1
              max_train_steps: 1
              save_steps: 1
              logging_steps: 1
              mixed_precision: fp16
            """
        )

        config = load_config(path)
        self.assertEqual(config["data"]["prompts"], ["one prompt"])
        self.assertEqual(config["data"]["prompt"], "one prompt")

    def test_load_config_from_namespace_applies_seed_zero(self):
        path = self._write_config(
            """
            data:
              prompts: ["one prompt"]
              root_dir: ./data
              num_pairs: 1
              num_frames: 8
              resolution: 256
            model:
              base_model: base
              motion_adapter: adapter
            training:
              seed: 42
              batch_size: 1
              gradient_accumulation_steps: 1
              max_train_steps: 1
              save_steps: 1
              logging_steps: 1
              mixed_precision: no
            """
        )
        args = argparse.Namespace(config=path, checkpoint=None, seed=0, num_frames=None)

        config = load_config_from_namespace(args)
        self.assertEqual(config["training"]["seed"], 0)

    def test_train_profile_requires_training_fields(self):
        path = self._write_config(
            """
            data:
              prompts: ["one prompt"]
              root_dir: ./data
            model:
              base_model: base
              motion_adapter: adapter
              lora_rank: 8
              lora_alpha: 16
            training:
              mixed_precision: fp16
            """
        )
        with self.assertRaisesRegex(ValueError, "training.seed is required"):
            load_config(path, profile="train")

    def test_get_prompt_list_and_default_prompt(self):
        config = {
            "data": {
                "prompts": ["first", "second"],
            }
        }
        self.assertEqual(get_prompt_list(config), ["first", "second"])
        self.assertEqual(get_default_prompt(config), "first")

    def test_resolve_prompt_override_prefers_explicit_prompt(self):
        config = {"data": {"prompts": ["first", "second"]}}
        self.assertEqual(
            resolve_prompt_override(config, prompt="custom prompt", prompt_index=None),
            "custom prompt",
        )

    def test_resolve_prompt_override_uses_prompt_index(self):
        config = {"data": {"prompts": ["first", "second"]}}
        self.assertEqual(
            resolve_prompt_override(config, prompt=None, prompt_index=1),
            "second",
        )

    def test_resolve_prompt_override_rejects_both_prompt_and_index(self):
        config = {"data": {"prompts": ["first"]}}
        with self.assertRaises(ValueError):
            resolve_prompt_override(config, prompt="x", prompt_index=0)

    def test_resolve_checkpoint_path_prefers_explicit_override(self):
        config = {"inference_checkpoint": "from-config"}
        self.assertEqual(resolve_checkpoint_path(config, "from-cli"), "from-cli")

    def test_resolve_checkpoint_path_falls_back_to_config(self):
        config = {"inference_checkpoint": "from-config"}
        self.assertEqual(resolve_checkpoint_path(config, None), "from-config")


if __name__ == "__main__":
    unittest.main()
