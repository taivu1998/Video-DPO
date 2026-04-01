import unittest
from unittest import mock

import torch

from src.devices import get_generator, resolve_torch_dtype


class DeviceTests(unittest.TestCase):
    def test_fp16_resolves_to_float16_on_mps(self):
        dtype = resolve_torch_dtype({"training": {"mixed_precision": "fp16"}}, device="mps")
        self.assertEqual(dtype, torch.float16)

    def test_fp16_falls_back_to_float32_on_cpu(self):
        dtype = resolve_torch_dtype({"training": {"mixed_precision": "fp16"}}, device="cpu")
        self.assertEqual(dtype, torch.float32)

    def test_bf16_resolves_on_supported_cuda(self):
        with mock.patch("torch.cuda.is_bf16_supported", return_value=True):
            dtype = resolve_torch_dtype({"training": {"mixed_precision": "bf16"}}, device="cuda")
        self.assertEqual(dtype, torch.bfloat16)

    def test_bf16_falls_back_when_unsupported(self):
        with mock.patch("torch.cuda.is_bf16_supported", return_value=False):
            dtype = resolve_torch_dtype({"training": {"mixed_precision": "bf16"}}, device="cuda")
        self.assertEqual(dtype, torch.float32)

    def test_get_generator_uses_cpu_generator_for_mps(self):
        generator = get_generator(7, "mps")
        self.assertEqual(generator.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
