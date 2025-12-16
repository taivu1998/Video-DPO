import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler
from tqdm import tqdm
import os
import logging
import gc

class DPOTrainer:
    """
    Implements Diffusion-DPO with Implicit Rewards for video diffusion.
    Objective: Maximize margin between Reference Error and Policy Error for Winner vs Loser.

    The DPO loss trains the policy model to better predict noise for "winner" samples
    (temporally consistent) while being worse at predicting noise for "loser" samples
    (temporally inconsistent/jittery).

    Supports two modes:
    - Standard mode: Uses a separate deep-copied reference model
    - Memory-optimized mode: Uses adapter toggling (ref_model=None)
    """
    def __init__(self, config, policy_model, ref_model, dataloader, accelerator, logger=None):
        self.config = config
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.dataloader = dataloader
        self.accelerator = accelerator
        self.beta = config['training']['beta']
        self.logger = logger or logging.getLogger(__name__)
        self.gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)

        # Memory-optimized mode: ref_model is None, use adapter toggling
        self.memory_optimized = ref_model is None
        if self.memory_optimized:
            self.logger.info("Using memory-optimized mode with adapter toggling")

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            config['model']['base_model'],
            subfolder="scheduler"
        )

        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config['training']['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        self.policy_model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.policy_model, self.optimizer, self.dataloader
        )

        # Only move ref_model if it exists (not memory-optimized mode)
        if self.ref_model is not None:
            self.ref_model.to(self.accelerator.device)

        # Create checkpoint directory
        os.makedirs(config['output_dir'], exist_ok=True)

    def _prepare_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Prepare latents for AnimateDiff UNet.
        Input shape from dataset: [B, C, F, H, W] (batch, channels, frames, height, width)
        AnimateDiff expects: [B, C, F, H, W] - same format, just ensure correct dtype
        """
        # Ensure float32 for training stability
        return latents.to(dtype=torch.float32)

    def _prepare_prompt_embeds(self, prompt_embeds: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        Prepare prompt embeddings for video generation.
        The prompt embeds need to be expanded for each frame or kept as-is depending on the model.

        Input shape: [B, seq_len, hidden_dim]
        Output shape: [B, seq_len, hidden_dim] (same, model handles frame expansion internally)
        """
        return prompt_embeds.to(dtype=torch.float32)

    def train(self):
        global_step = 0
        save_dir = self.config['output_dir']
        logging_steps = self.config['training'].get('logging_steps', 10)
        max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)

        self.policy_model.train()
        self.logger.info(f"Starting training for {self.config['training']['max_train_steps']} steps")
        self.logger.info(f"Batch size: {self.config['training']['batch_size']}, "
                        f"Gradient accumulation: {self.gradient_accumulation_steps}")
        self.logger.info(f"Beta (DPO scalar): {self.beta}")

        # Track running loss and metrics for logging
        running_loss = 0.0
        running_reward_w = 0.0
        running_reward_l = 0.0
        num_batches = 0

        # Infinite loop until max_steps
        epoch = 0
        while global_step < self.config['training']['max_train_steps']:
            epoch += 1
            progress_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch}",
                disable=not self.accelerator.is_local_main_process
            )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.policy_model):
                    # Unpack and prepare tensors
                    # Dataset provides: [C, F, H, W] per sample, DataLoader adds batch dim: [B, C, F, H, W]
                    latents_w = self._prepare_latents(batch['latents_w'])
                    latents_l = self._prepare_latents(batch['latents_l'])
                    prompt_embeds = self._prepare_prompt_embeds(
                        batch['prompt_embeds'],
                        num_frames=latents_w.shape[2]  # F dimension
                    )

                    bsz = latents_w.shape[0]

                    # 1. Sample Noise (same noise for winner and loser for fair comparison)
                    noise = torch.randn_like(latents_w)

                    # Sample random timesteps for each batch element
                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps, (bsz,),
                        device=self.accelerator.device
                    ).long()

                    # 2. Add Noise (Forward Diffusion Process)
                    # This creates noisy versions of both winner and loser latents
                    noisy_w = self.noise_scheduler.add_noise(latents_w, noise, timesteps)
                    noisy_l = self.noise_scheduler.add_noise(latents_l, noise, timesteps)

                    # 3. Reference Model Forward Pass (Frozen, No Gradients)
                    with torch.no_grad():
                        if self.memory_optimized:
                            # Disable LoRA adapters to get reference model behavior
                            self.accelerator.unwrap_model(self.policy_model).disable_adapter_layers()

                            ref_pred_w = self.policy_model(
                                noisy_w, timesteps, encoder_hidden_states=prompt_embeds
                            ).sample
                            ref_pred_l = self.policy_model(
                                noisy_l, timesteps, encoder_hidden_states=prompt_embeds
                            ).sample

                            # Re-enable LoRA adapters for policy model
                            self.accelerator.unwrap_model(self.policy_model).enable_adapter_layers()
                        else:
                            ref_pred_w = self.ref_model(
                                noisy_w, timesteps, encoder_hidden_states=prompt_embeds
                            ).sample
                            ref_pred_l = self.ref_model(
                                noisy_l, timesteps, encoder_hidden_states=prompt_embeds
                            ).sample

                    # 4. Policy Model Forward Pass (Trainable LoRA)
                    policy_pred_w = self.policy_model(
                        noisy_w, timesteps, encoder_hidden_states=prompt_embeds
                    ).sample
                    policy_pred_l = self.policy_model(
                        noisy_l, timesteps, encoder_hidden_states=prompt_embeds
                    ).sample

                    # 5. Compute Implicit Rewards
                    # Reward = Reference_Error - Policy_Error
                    # Positive reward = policy is better than reference at denoising
                    # We reduce over all dims except batch [C, F, H, W] -> scalar per sample

                    # Compute MSE errors
                    ref_error_w = (ref_pred_w - noise).pow(2)
                    ref_error_l = (ref_pred_l - noise).pow(2)
                    policy_error_w = (policy_pred_w - noise).pow(2)
                    policy_error_l = (policy_pred_l - noise).pow(2)

                    # Reduce over spatial/temporal dims, keep batch dim
                    reduce_dims = list(range(1, len(ref_error_w.shape)))
                    reward_w = ref_error_w.mean(dim=reduce_dims) - policy_error_w.mean(dim=reduce_dims)
                    reward_l = ref_error_l.mean(dim=reduce_dims) - policy_error_l.mean(dim=reduce_dims)

                    # 6. DPO Loss
                    # We want: reward_w > reward_l (policy better on winner, worse on loser)
                    # Loss = -log(sigmoid(beta * (reward_w - reward_l)))
                    reward_margin = reward_w - reward_l
                    loss = -F.logsigmoid(self.beta * reward_margin).mean()

                    # Backward pass
                    self.accelerator.backward(loss)

                    # Gradient clipping (only when gradients are synchronized)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.policy_model.parameters(), max_grad_norm)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Track metrics
                    running_loss += loss.detach().item()
                    running_reward_w += reward_w.mean().detach().item()
                    running_reward_l += reward_l.mean().detach().item()
                    num_batches += 1

                # Only increment global_step after gradient sync (accumulation complete)
                if self.accelerator.sync_gradients:
                    global_step += 1

                    # Clear memory periodically in memory-optimized mode
                    if self.memory_optimized and global_step % 10 == 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        gc.collect()

                    # Logging
                    if global_step % logging_steps == 0:
                        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
                        avg_rw = running_reward_w / num_batches if num_batches > 0 else 0.0
                        avg_rl = running_reward_l / num_batches if num_batches > 0 else 0.0

                        self.logger.info(
                            f"Step {global_step}/{self.config['training']['max_train_steps']} - "
                            f"Loss: {avg_loss:.6f}, Reward_W: {avg_rw:.6f}, Reward_L: {avg_rl:.6f}"
                        )
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "r_w": f"{avg_rw:.4f}",
                            "r_l": f"{avg_rl:.4f}",
                            "step": global_step
                        })

                        running_loss = 0.0
                        running_reward_w = 0.0
                        running_reward_l = 0.0
                        num_batches = 0

                    # Save checkpoint
                    if global_step % self.config['training']['save_steps'] == 0:
                        self._save_checkpoint(save_dir, global_step)

                    if global_step >= self.config['training']['max_train_steps']:
                        break

            if global_step >= self.config['training']['max_train_steps']:
                break

        # Save final checkpoint
        self._save_checkpoint(save_dir, global_step)
        self.logger.info(f"Training complete! Final step: {global_step}")

    def _save_checkpoint(self, save_dir, step):
        path = os.path.join(save_dir, f"checkpoint-{step}")
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(self.policy_model)
            unwrapped.save_pretrained(path)
            self.logger.info(f"Saved checkpoint to {path}")

            # Also save as 'latest' for easy inference
            latest_path = os.path.join(save_dir, "latest")
            if os.path.exists(latest_path):
                import shutil
                shutil.rmtree(latest_path)
            unwrapped.save_pretrained(latest_path)
            self.logger.info(f"Updated latest checkpoint")