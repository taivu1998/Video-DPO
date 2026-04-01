import torch
import gc
from diffusers import MotionAdapter, AnimateDiffPipeline
from diffusers.schedulers import DDIMScheduler
from peft import LoraConfig, get_peft_model, PeftModel
import copy

from src.devices import get_device, resolve_torch_dtype

class VideoDPOModelWrapper:
    def __init__(self, config: dict):
        self.config = config
        # Memory-optimized mode uses adapter toggling instead of deep copy
        self.memory_optimized = config.get('training', {}).get('memory_optimized', False)

    def _resolve_model_dtype(self, device=None) -> torch.dtype:
        return resolve_torch_dtype(self.config, device=device or get_device())

    def _get_motion_module_names(self, model) -> list:
        """
        Get the names of all motion module attention layers for LoRA targeting.
        Only targets to_q, to_k, to_v, to_out.0 within motion_modules.
        """
        target_suffixes = ["to_q", "to_k", "to_v", "to_out.0"]
        motion_module_names = []

        for name, module in model.named_modules():
            # Check if this is a motion module attention layer
            if "motion_modules" in name:
                for suffix in target_suffixes:
                    if name.endswith(suffix):
                        motion_module_names.append(name)
                        break

        return motion_module_names

    def get_trainable_model(self):
        """
        Prepares Policy (LoRA) and Reference models.

        If memory_optimized=True (from config), uses adapter toggling instead of
        deep copying the reference model. This saves ~50% GPU memory but requires
        the trainer to use disable_adapter_layers()/enable_adapter_layers().

        Returns:
            policy_model: UNet with LoRA adapters
            ref_model: Either a deep copy (memory_optimized=False) or None (memory_optimized=True)
        """
        # Clear memory before loading
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        print("Loading Motion Adapter...")
        dtype = self._resolve_model_dtype()
        adapter = MotionAdapter.from_pretrained(
            self.config['model']['motion_adapter'],
            torch_dtype=dtype
        )

        print("Loading AnimateDiff Pipeline...")
        # 2. Load full AnimateDiff pipeline to get properly configured UNet
        pipe = AnimateDiffPipeline.from_pretrained(
            self.config['model']['base_model'],
            motion_adapter=adapter,
            torch_dtype=dtype
        )

        # 3. Extract UNet
        unet_policy = pipe.unet

        # 4. Create reference model based on memory mode
        if self.memory_optimized:
            print("Memory-optimized mode: Using adapter toggling (no deep copy)")
            unet_ref = None  # Will use adapter toggling in trainer
        else:
            print("Creating reference model (deep copy)...")
            # Deep copy for reference model (must be done before LoRA injection)
            unet_ref = copy.deepcopy(unet_policy)
            unet_ref.requires_grad_(False)
            unet_ref.eval()

        # 5. Freeze policy base weights
        unet_policy.requires_grad_(False)

        # 6. Find motion module layers to target
        print("Finding motion module layers for LoRA...")
        motion_module_names = self._get_motion_module_names(unet_policy)

        if not motion_module_names:
            # Fallback: use regex pattern matching
            print("Using regex pattern matching for LoRA targets...")
            # PEFT supports regex in target_modules
            target_modules = r".*motion_modules.*(to_q|to_k|to_v|to_out\.0)$"
        else:
            print(f"Found {len(motion_module_names)} motion module layers to target")
            target_modules = motion_module_names

        # 7. Inject LoRA into policy model
        lora_config = LoraConfig(
            r=self.config['model']['lora_rank'],
            lora_alpha=self.config['model']['lora_alpha'],
            target_modules=target_modules,
            lora_dropout=0.0 if self.memory_optimized else 0.1,
            bias="none",
            modules_to_save=None,
        )

        print("Injecting LoRA adapters...")
        unet_policy = get_peft_model(unet_policy, lora_config)
        unet_policy.print_trainable_parameters()

        # Enable gradient checkpointing for memory efficiency
        if self.memory_optimized and hasattr(unet_policy, 'enable_gradient_checkpointing'):
            unet_policy.enable_gradient_checkpointing()
            print("Gradient checkpointing enabled")

        # Verify we're only training motion modules
        trainable_params = [n for n, p in unet_policy.named_parameters() if p.requires_grad]
        motion_params = [n for n in trainable_params if "motion_modules" in n]
        non_motion_params = [n for n in trainable_params if "motion_modules" not in n]

        if non_motion_params:
            print(f"WARNING: {len(non_motion_params)} non-motion parameters are trainable!")
            print("First few:", non_motion_params[:5])
        else:
            print(f"SUCCESS: All {len(motion_params)} trainable parameters are in motion_modules")

        # Clean up pipeline to free memory (we only need the UNets)
        del pipe
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        if torch.cuda.is_available():
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        return unet_policy, unet_ref

    def get_inference_pipeline(self, device, lora_path=None):
        """Load inference pipeline with optional LoRA weights."""
        dtype = self._resolve_model_dtype(device)
        adapter = MotionAdapter.from_pretrained(
            self.config['model']['motion_adapter'],
            torch_dtype=dtype
        )
        pipe = AnimateDiffPipeline.from_pretrained(
            self.config['model']['base_model'],
            motion_adapter=adapter,
            torch_dtype=dtype
        )

        # Use DDIM for faster inference
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        if lora_path:
            print(f"Loading LoRA weights from {lora_path}...")
            # Load the trained DPO LoRA adapter
            pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
            pipe.unet = pipe.unet.merge_and_unload()  # Merge for faster inference

        pipe.to(device)
        pipe.enable_vae_slicing()  # Memory optimization

        return pipe
