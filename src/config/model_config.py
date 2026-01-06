"""Model-level bit-width configuration."""
from dataclasses import dataclass
from typing import List
from .layer_config import LayerBitWidthConfig


@dataclass
class ModelBitWidthConfig:
    """
    Bit-width configuration for entire model (all 12 layers).
    """
    layer_configs: List[LayerBitWidthConfig]
    
    def __getitem__(self, idx: int) -> LayerBitWidthConfig:
        """Allow indexing: config[0] returns layer 0's config."""
        return self.layer_configs[idx]
    
    def __len__(self) -> int:
        """Return number of layers."""
        return len(self.layer_configs)
    

def create_32bit_model_config() -> ModelBitWidthConfig:
    """
    Create a 32-bit ModelBitWidthConfig (no quantization).
    
    All 12 layers use 32-bit precision for:
    - Weights (bit_width_w=32)
    - Activations (bit_width_a=32)
    - KV cache (bit_width_kv=32)
    
    This config is useful for:
    1. Loading pretrained weights without quantization
    2. Verifying outputs match original GPT-2
    3. Baseline accuracy comparison
    
    Returns:
        ModelBitWidthConfig with all layers at 32-bit
    
    Example:
        >>> config_32bit = create_32bit_model_config()
    """
    return ModelBitWidthConfig([
        LayerBitWidthConfig(
            bit_width_w=32,
            bit_width_a=32,
            bit_width_kv=32
        )
        for _ in range(12)  # 12 transformer blocks
    ])

def create_uniform_model_config(
    bit_width_w: int,
    bit_width_a: int = 16,
    bit_width_kv: int = 32,
    num_layers: int = 12
) -> ModelBitWidthConfig:
    """
    Create a uniform bit-width configuration for all layers.
    
    All layers will have the same quantization bit-widths.
    
    Args:
        bit_width_w: Weight quantization bit-width (e.g., 4, 6, 8, 12, 16)
        bit_width_a: Activation quantization bit-width (default: 16)
        bit_width_kv: KV cache quantization bit-width (default: 32)
        num_layers: Number of transformer layers (default: 12 for GPT-2)
    
    Returns:
        ModelBitWidthConfig with all layers having the same bit-widths
    """
    layer_configs = [
        LayerBitWidthConfig(
            bit_width_w=bit_width_w,
            bit_width_a=bit_width_a,
            bit_width_kv=bit_width_kv
        )
        for _ in range(num_layers)
    ]
    return ModelBitWidthConfig(layer_configs)