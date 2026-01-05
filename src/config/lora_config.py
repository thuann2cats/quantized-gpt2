"""LoRA (Low-Rank Adaptation) configuration."""
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA adapters.
    """
    r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1