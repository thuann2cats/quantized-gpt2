"""
Configuration loader for training configs.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

from src.config.layer_config import LayerBitWidthConfig
from src.config.model_config import ModelBitWidthConfig, create_uniform_model_config
from src.config.lora_config import LoRAConfig


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


def load_training_config(config_path: str) -> Dict[str, Any]:
    """
    Load training configuration from YAML file as a dict.
    Parses bit-width configs and LoRA config into proper objects.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Config dict with YAML hierarchy preserved
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Parse bit-width configs (if present - for joint training)
    if 'bit_width_configs' in config['model']:
        bit_width_configs = {}
        for name, cfg in config['model']['bit_width_configs'].items():
            bit_width_configs[name] = parse_bit_width_config(cfg)
        config['model']['bit_width_configs'] = bit_width_configs
    
    # Parse cyclic_schedule bit-width configs (if present - for cyclic training)
    if 'cyclic_schedule' in config['model']:
        schedule = config['model']['cyclic_schedule']
        # Generate bit-width configs for all bit widths in range
        bit_width_configs = {}
        for bw in range(schedule['b_min'], schedule['b_max'] + 1):
            bit_width_configs[f'uniform_{bw}bit'] = create_uniform_model_config(
                bit_width_w=bw,
                bit_width_a=schedule.get('bit_width_a', 16),
                bit_width_kv=schedule.get('bit_width_kv', 32)
            )
        config['model']['bit_width_configs'] = bit_width_configs
    
    # Parse LoRA config
    if 'lora' in config:
        config['lora'] = LoRAConfig(**config['lora'])
    
    return config


# Convenience functions
def load_joint_training_config() -> Dict[str, Any]:
    """Load default joint training config."""
    return load_training_config('configs/training/joint_training.yaml')


def load_cyclic_training_config() -> Dict[str, Any]:
    """Load default cyclic precision training config."""
    return load_training_config('configs/training/cyclic_precision.yaml')
