"""
Training script for Baseline GPT-2 Question Answering.

Usage:
    python scripts/train_baseline.py --config configs/training/baseline_training.yaml
"""
from src.utils import env_setup

import argparse
import os
import shutil
import torch
import random
import numpy as np
from transformers import GPT2Config

# Import our modules
from src.config.config_loader import load_training_config
from src.data.squad_dataset import prepare_squad_data
from transformers import GPT2ForQuestionAnswering
from src.utils.weight_loading import load_pretrained_gpt2_weights
from src.training.baseline_trainer import BaselineTrainer


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train Baseline GPT-2 for Question Answering"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training config YAML file'
    )
    
    args = parser.parse_args()
    
    # Load config
    print("=" * 80)
    print("Loading configuration...")
    print("=" * 80)
    config = load_training_config(args.config)
    
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Type: {config['experiment']['type']}")
    print(f"Device: {config['training']['device']}")
    
    # Set seed
    set_seed(config['training']['seed'])
    print(f"Random seed set to: {config['training']['seed']}")
    
    # Prepare data
    print("\n" + "=" * 80)
    print("Preparing SQuAD data...")
    print("=" * 80)
    train_loader, val_loader, tokenizer = prepare_squad_data(
        dataset_name=config['data']['dataset_name'],
        batch_size=config['training']['batch_size'],
        max_length=config['data']['max_length'],
        val_size=config['data']['val_split_ratio']
    )
    
    # Create model
    print("\n" + "=" * 80)
    print("Creating model...")
    print("=" * 80)
    
    # Create baseline model using HuggingFace's implementation
    print("\n" + "=" * 80)
    print(f"Loading pretrained model: {config['model']['pretrained_model']}")
    print("=" * 80)
    
    model = GPT2ForQuestionAnswering.from_pretrained(config['model']['pretrained_model'])
    print("✅ Baseline GPT-2 Model loaded (No Quantization, No LoRA)")
    
    # Create trainer
    print("\n" + "=" * 80)
    print("Creating trainer...")
    print("=" * 80)
    
    trainer = BaselineTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer
    )
    print("✅ Baseline Trainer created")
    
    # Copy config to experiment directory
    config_copy_path = os.path.join(trainer.experiment_dir, 'config.yaml')
    shutil.copy(args.config, config_copy_path)
    print(f"✅ Config copied to: {config_copy_path}")
    
    # Start training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    try:
        trainer.train()
        print("\n" + "=" * 80)
        print("✅ Training completed successfully!")
        print("=" * 80)
        print(f"Experiment directory: {trainer.experiment_dir}")
        
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("⚠️  Training interrupted by user")
        print("=" * 80)
        print(f"Partial results saved in: {trainer.experiment_dir}")
    
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ Training failed with error:")
        print("=" * 80)
        print(str(e))
        raise


if __name__ == "__main__":
    main()
