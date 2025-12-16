import yaml
import argparse
import os
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Loads YAML and overrides with CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint for inference")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
        
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # CLI Overrides
    if args.checkpoint:
        config['inference_checkpoint'] = args.checkpoint
    if args.seed:
        config['training']['seed'] = args.seed
        
    return config