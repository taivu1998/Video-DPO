.PHONY: install setup-accelerate clean data train inference evaluate all test-imports test-data test-train test-all validate-data

# ============================================
# INSTALLATION
# ============================================

# Full installation
install:
	pip install -r requirements.txt
	pip install -e .

# Setup accelerate (run once before training)
setup-accelerate:
	accelerate config --config_file configs/accelerate_config.yaml

# Test that all imports work
test-imports:
	python -c "from src.model import VideoDPOModelWrapper; from src.trainer import DPOTrainer; from src.dataset import VideoDPODataset; print('All imports successful!')"

# ============================================
# FULL PIPELINE (Production)
# ============================================

# Phase 1: Data Gen (Expect ~3 hours on A100)
data:
	python scripts/generate_data.py --config configs/train_config.yaml

# Phase 2: Training (Expect ~12 hours)
train:
	accelerate launch --config_file configs/accelerate_config.yaml scripts/train.py --config configs/train_config.yaml

# Alternative: Train without accelerate config (auto-detect)
train-auto:
	accelerate launch scripts/train.py --config configs/train_config.yaml

# Phase 3: Validation (Side-by-side GIF comparison)
inference:
	python scripts/inference.py --config configs/train_config.yaml --checkpoint checkpoints/latest

# Phase 4: Quantitative Evaluation (Warping Error Metric)
evaluate:
	python scripts/evaluate.py --config configs/train_config.yaml --checkpoint checkpoints/latest --num_samples 20

# Run full pipeline
all: data train inference evaluate

# Validate generated data
validate-data:
	python scripts/validate_data.py --config configs/train_config.yaml

# ============================================
# TESTING (Quick validation with minimal data)
# ============================================

# Generate test data (5 pairs only)
test-data:
	python scripts/generate_data.py --config configs/test_config.yaml

# Validate test data
test-validate:
	python scripts/validate_data.py --config configs/test_config.yaml

# Run test training (20 steps)
test-train:
	accelerate launch scripts/train.py --config configs/test_config.yaml

# Run test inference
test-inference:
	python scripts/inference.py --config configs/test_config.yaml --checkpoint checkpoints_test/latest

# Full test pipeline
test-all: test-data test-validate test-train test-inference
	@echo "Test pipeline completed successfully!"

# ============================================
# CLEANUP
# ============================================

# Clean all generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf logs/ checkpoints/ data/ evaluation_results/

# Clean test files only
clean-test:
	rm -rf logs_test/ checkpoints_test/ data/latents_test/