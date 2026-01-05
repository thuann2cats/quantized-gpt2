"""Layer-level bit-width configuration"""
from dataclasses import dataclass


@dataclass
class LayerBitWidthConfig:
    """Bit-width configuration for a single transformer layer."""
    bit_width_w: int
    bit_width_a: int
    bit_width_kv: int