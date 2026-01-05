import torch
import torch.nn as nn
from typing import Dict, Optional, Union
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from src.models.quantized_attention import QuantizedGPT2Attention
from src.models.quantized_mlp import QuantizedGPT2MLP
from src.config.layer_config import LayerBitWidthConfig
from src.config.lora_config import LoRAConfig


class QuantizedGPT2Block(GPT2Block):
    """
    Quantized version of GPT2Block.
    
    Inherits from GPT2Block and replaces:
    1. attn: GPT2Attention -> QuantizedGPT2Attention
    2. mlp: GPT2MLP -> QuantizedGPT2MLP
    
    Keeps:
    - LayerNorm layers (ln_1, ln_2)
    - forward() logic with residual connections
    - Optional cross-attention support
    """

    def __init__(
        self,
        config,
        available_layer_bit_width_configs: Dict[str, LayerBitWidthConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        layer_idx: Optional[int] = None
    ):
        # Call parent to set up LayerNorms and create attn/mlp
        # Parent creates: ln_1, attn, ln_2, mlp (and optionally crossattention, ln_cross_attn)
        super().__init__(config, layer_idx=layer_idx)

        # Store configs for later use
        self.available_layer_bit_width_configs = available_layer_bit_width_configs or {}
        self.current_layer_bit_width_config: Optional[str] = None

        # REPLACE attn with QuantizedGPT2Attention
        del self.attn
        self.attn = QuantizedGPT2Attention(
            config=config,
            available_layer_bit_width_configs=available_layer_bit_width_configs,
            lora_config=lora_config,
            is_cross_attention=False,
            layer_idx=layer_idx
        )

        # REPLACE mlp with QuantizedGPT2MLP
        del self.mlp
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
        self.mlp = QuantizedGPT2MLP(
            intermediate_size=inner_dim,
            config=config,
            available_layer_bit_width_configs=available_layer_bit_width_configs,
            lora_config=lora_config
        )

        # Handle cross-attention (if enabled)
        if config.add_cross_attention:
            # Parent created self.crossattention and self.ln_cross_attn
            # Replace crossattention with quantized version
            if hasattr(self, 'crossattention'):
                del self.crossattention
                self.crossattention = QuantizedGPT2Attention(
                    config=config,
                    available_layer_bit_width_configs=available_layer_bit_width_configs,
                    lora_config=lora_config,
                    is_cross_attention=True,
                    layer_idx=layer_idx
                )

    def set_layer_bit_width_config(self, config_name_or_obj: Union[str, LayerBitWidthConfig]):
        """
        Set quantization configuration for all sub-modules (attn, mlp, crossattention)
        """
        # Update current config
        if isinstance(config_name_or_obj, str):
            if config_name_or_obj not in self.available_layer_bit_width_configs:
                raise ValueError(f"Config '{config_name_or_obj}' not found in available configs")
                self.current_layer_bit_width_config = config_name_or_obj

        else:
            self.current_layer_bit_width_config = "custom"

        # Propagate to sub-modules
        self.attn.set_layer_bit_width_config(config_name_or_obj)
        self.mlp.set_layer_bit_width_config(config_name_or_obj)

        # Also set for cross-attention if it exists
        if hasattr(self, 'crossattention'):
            self.crossattention.set_layer_bit_width_config(config_name_or_obj)

    # Note: forward() is inherited from GPT2Block and works unchanged
    # The inherited forward() calls self.attn and self.mlp, which are now quantized versions