#!/usr/bin/env python3
"""
Simple script to run the fine-tuning pipeline with example data.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config
from src.train import LLMFineTuner


def main():
    """Run the fine-tuning example."""
    print("="*60)
    print("LLM Fine-Tuning Example")
    print("="*60)
    print()
    
    # Load configuration
    config_path = "config.yaml"
    print(f"Loading configuration from {config_path}...")
    config = Config.from_yaml(config_path)
    
    print(f"Model: {config.model.name}")
    print(f"Training epochs: {config.training.num_train_epochs}")
    print(f"Output directory: {config.training.output_dir}")
    print()
    
    # Run fine-tuning
    fine_tuner = LLMFineTuner(config)
    fine_tuner.run_full_pipeline()
    
    print()
    print("="*60)
    print("Fine-tuning completed successfully!")
    print(f"Model saved to: {config.training.output_dir}")
    print()
    print("To run inference, use:")
    print(f"  python src/inference.py --model_path {config.training.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
