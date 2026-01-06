"""
Configuration loader for training configs.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

from src.config.layer_config import LayerBitWidthConfig
from src.config.model_config import ModelBitWidthConfig, create_uniform_model_config
from src.config.lora_config import LoRAConfig


@dataclass
class TrainingConfig:
    """Parsed training configuration."""
    
    # Experiment
    experiment_name: str
    experiment_type: str  # "joint" or "cyclic"
    base_folder: str
    description: str
    
    # Model
    pretrained_model: str
    bit_width_configs: Dict[str, ModelBitWidthConfig]
    
    # Training
    num_steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_grad_norm: float
    log_every_steps: int
    validate_every_steps: int
    num_validation_steps: int
    optimizer: str
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    device: str
    seed: int
    
    # CPT-specific (None for joint training)
    cyclic_schedule: Dict[str, Any] = None
    
    # LoRA
    lora_config: LoRAConfig = None
    
    # Data
    dataset_name: str = "squad_v2"
    max_length: int = 384
    doc_stride: int = 128
    val_split_ratio: float = 0.1
    val_split_seed: int = 42
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "quantized-gpt2-qa"
    wandb_entity: str = None
    log_to_file: bool = True
    
    # Checkpointing
    save_final: bool = True
    checkpoint_name: str = "final_checkpoint.pt"


def parse_bit_width_config(config_dict: Dict[str, Any]) -> ModelBitWidthConfig:
    """
    Parse a single bit-width config from YAML.
    
    Supports two formats:
    1. Uniform: {bit_width_w: X, bit_width_a: Y, bit_width_kv: Z} - all layers same
    2. Explicit: {layers: [{bit_width_w:..., bit_width_a:..., bit_width_kv:...}, ...]} - per-layer
    
    Args:
        config_dict: Dict with bit-width fields or 'layers' key
    
    Returns:
        ModelBitWidthConfig
    """
    # Check if explicit per-layer specification
    if 'layers' in config_dict:
        layer_configs = [
            LayerBitWidthConfig(**layer) for layer in config_dict['layers']
        ]
        return ModelBitWidthConfig(layer_configs)
    
    # Otherwise, uniform config - use helper function
    return create_uniform_model_config(
        bit_width_w=config_dict['bit_width_w'],
        bit_width_a=config_dict['bit_width_a'],
        bit_width_kv=config_dict['bit_width_kv']
    )


def load_training_config(config_path: str) -> TrainingConfig:
    """
    Load training configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        TrainingConfig object
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Parse bit-width configs
    bit_width_configs = {}
    for name, cfg in config['model']['bit_width_configs'].items():
        bit_width_configs[name] = parse_bit_width_config(cfg)
    
    # Parse LoRA config
    lora_config = LoRAConfig(**config['lora'])
    
    # Build TrainingConfig
    training_config = TrainingConfig(
        # Experiment
        experiment_name=config['experiment']['name'],
        experiment_type=config['experiment']['type'],
        base_folder=config['experiment']['base_folder'],
        description=config['experiment']['description'],
        
        # Model
        pretrained_model=config['model']['pretrained_model'],
        bit_width_configs=bit_width_configs,
        
        # Training
        num_steps=config['training']['num_steps'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        log_every_steps=config['training']['log_every_steps'],
        validate_every_steps=config['training']['validate_every_steps'],
        num_validation_steps=config['training']['num_validation_steps'],
        optimizer=config['training']['optimizer'],
        adam_beta1=config['training']['adam_beta1'],
        adam_beta2=config['training']['adam_beta2'],
        adam_epsilon=config['training']['adam_epsilon'],
        device=config['training']['device'],
        seed=config['training']['seed'],
        
        # CPT-specific (may be None)
        cyclic_schedule=config['training'].get('cyclic_schedule', None),
        
        # LoRA
        lora_config=lora_config,
        
        # Data
        dataset_name=config['data']['dataset_name'],
        max_length=config['data']['max_length'],
        doc_stride=config['data']['doc_stride'],
        val_split_ratio=config['data']['val_split_ratio'],
        val_split_seed=config['data']['val_split_seed'],
        
        # Logging
        use_wandb=config['logging']['use_wandb'],
        wandb_project=config['logging']['wandb_project'],
        wandb_entity=config['logging']['wandb_entity'],
        log_to_file=config['logging']['log_to_file'],
        
        # Checkpointing
        save_final=config['checkpointing']['save_final'],
        checkpoint_name=config['checkpointing']['checkpoint_name'],
    )
    
    return training_config


# Convenience function
def load_joint_training_config() -> TrainingConfig:
    """Load default joint training config."""
    return load_training_config('configs/training/joint_training.yaml')


def load_cyclic_training_config() -> TrainingConfig:
    """Load default cyclic precision training config."""
    return load_training_config('configs/training/cyclic_precision.yaml')