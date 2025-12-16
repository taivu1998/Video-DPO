#!/bin/bash
# Video-DPO Setup Script
# This script sets up the environment for Video-DPO training

set -e  # Exit on error

echo "=============================================="
echo "Video-DPO Setup Script"
echo "=============================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION < 3.9" | bc -l) -eq 1 ]]; then
    echo "ERROR: Python 3.9+ is required"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (detect platform)
echo ""
echo "Installing PyTorch..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - use MPS
    echo "Detected macOS - installing PyTorch with MPS support"
    pip install torch torchvision torchaudio
elif command -v nvidia-smi &> /dev/null; then
    # NVIDIA GPU available
    echo "Detected NVIDIA GPU - installing PyTorch with CUDA support"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    # CPU only
    echo "No GPU detected - installing CPU-only PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Install package in editable mode
echo ""
echo "Installing video-dpo package..."
pip install -e .

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS available: {torch.backends.mps.is_available()}')

import diffusers
print(f'Diffusers: {diffusers.__version__}')

import transformers
print(f'Transformers: {transformers.__version__}')

import accelerate
print(f'Accelerate: {accelerate.__version__}')

import peft
print(f'PEFT: {peft.__version__}')

from src.model import VideoDPOModelWrapper
from src.trainer import DPOTrainer
from src.dataset import VideoDPODataset
print('')
print('All imports successful!')
"

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Generate training data: make data"
echo "  3. Start training: make train"
echo "  4. Run inference: make inference"
echo ""
echo "For a quick test with fewer samples, edit configs/train_config.yaml"
echo "and reduce num_pairs to 10-20 first."
echo ""
