import sys
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from src.config_parser import load_config
from src.utils import seed_everything, setup_logger
from src.dataset import VideoDPODataset
from src.model import VideoDPOModelWrapper
from src.trainer import DPOTrainer

def main():
    config = load_config()

    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config.get('log_dir', './logs'), exist_ok=True)

    logger = setup_logger("VideoDPO", config.get('log_dir', './logs'))
    seed_everything(config['training']['seed'])

    logger.info("="*50)
    logger.info("Video-DPO Training")
    logger.info("="*50)
    logger.info(f"Experiment: {config.get('experiment_name', 'video_dpo')}")
    logger.info(f"Output dir: {config['output_dir']}")
    logger.info(f"Data dir: {config['data']['root_dir']}")

    # Initialize accelerator with gradient accumulation
    accelerator = Accelerator(
        mixed_precision=config['training']['mixed_precision'],
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1)
    )

    logger.info(f"Device: {accelerator.device}")
    logger.info(f"Mixed precision: {config['training']['mixed_precision']}")
    logger.info(f"Gradient accumulation steps: {config['training'].get('gradient_accumulation_steps', 1)}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = VideoDPODataset(config['data']['root_dir'])
    logger.info(f"Dataset size: {len(dataset)} pairs")

    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Load models
    logger.info("Loading models...")
    wrapper = VideoDPOModelWrapper(config)
    policy_model, ref_model = wrapper.get_trainable_model()
    logger.info("Models loaded successfully")

    # Initialize trainer and start training
    trainer = DPOTrainer(
        config=config,
        policy_model=policy_model,
        ref_model=ref_model,
        dataloader=dataloader,
        accelerator=accelerator,
        logger=logger
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed!")

if __name__ == "__main__":
    main()