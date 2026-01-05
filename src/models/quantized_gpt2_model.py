import torch
import torch.nn as nn
from typing import Dict, Optional, Union
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

from src.models.quantized_block import QuantizedGPT2Block
from src.config.model_config import ModelBitWidthConfig
from src.config.lora_config import LoRAConfig


class QuantizedGPT2Model(GPT2Model):
    """
    Quantized version of GPT2Model.
    
    Inherits from transformers' GPT2Model and replaces the 12 GPT2Blocks
    with 12 QuantizedGPT2Blocks.
    
    Keeps unchanged:
    - wte: token embeddings
    - wpe: position embeddings
    - ln_f: final layer normalization
    - forward() method (inherited)
    
    Replaces:
    - h: ModuleList of GPT2Block → ModuleList of QuantizedGPT2Block
    
    Adds:
    - available_model_bit_width_configs: Dict[str, ModelBitWidthConfig]
    - current_model_bit_width_config: Optional[str]
    - set_model_bit_width_config(): method to switch bit-width configs
    """

    def __init__(
        self,
        config,
        available_model_bit_width_configs: Dict[str, ModelBitWidthConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ):
        """
        Initialize QuantizedGPT2Model.
        
        Args:
            config: GPT2Config or QuantizedGPT2Config
            available_model_bit_width_configs: Dict mapping config names to ModelBitWidthConfig
                Example: {
                    '2bit': ModelBitWidthConfig([...12 layer configs...]),
                    '4bit': ModelBitWidthConfig([...12 layer configs...]),
                }
            lora_config: LoRAConfig for all LoRA adapters
        """

        # Initialize parent class (creates wte, wpe, h, ln_f, etc.)
        super().__init__(config)

        # Store model-level configs
        self.available_model_bit_width_configs = available_model_bit_width_configs or {}
        self.current_model_bit_width_config: Optional[str] = None

        # Delete the original blocks created by parent
        # (These are GPT2Block instances, we need QuantizedGPT2Block)
        del self.h

        # Create new ModuleList with QuantizedGPT2Blocks
        self.h = nn.ModuleList()

        # Create 12 quantized blocks (or config.num_hidden_layers)
        for i in range(config.num_hidden_layers):
            # Extract layer-specific configs from all model configs
            # This is the KEY operation!
            layer_configs = {
                config_name: model_bit_width_config[i]
                for config_name, model_bit_width_config
                in self.available_model_bit_width_configs.items()
            }
        
            # Create quantized block for this layer
            block = QuantizedGPT2Block(
                config=config,
                available_layer_bit_width_configs=layer_configs,
                lora_config=lora_config,
                layer_idx=i
            )

            self.h.append(block)


    def set_model_bit_width_config(
        self,
        config_name_or_obj: Union[str, ModelBitWidthConfig]
    ):
        """
        Set the bit-width configuration for the entire model.
        
        This cascades down to all blocks, which cascade to all layers.
        
        Args:
            config_name_or_obj: Either:
                - str: name of config in available_model_bit_width_configs (e.g., '4bit')
                - ModelBitWidthConfig: custom config object (list of 12 LayerBitWidthConfigs)
        
        Example:
            model.set_model_bit_width_config('4bit')
            # Now all layers use 4-bit weights
            
            model.set_model_bit_width_config('mixed')
            # Now layers use mixed bit-widths as defined in 'mixed' config
        """
        # handle both string names and direct config objects
        if isinstance(config_name_or_obj, str):
            config_name = config_name_or_obj

            # Validate config name exists
            if config_name not in self.available_model_bit_width_configs:
                available = list(self.available_model_bit_width_configs.keys())
                raise ValueError(
                    f"Config '{config_name}' not found. "
                    f"Available configs: {available}"
                )
            
            # Update current config name
            self.current_model_bit_width_config = config_name

            # Cascade to all blocks
            for i, block in enumerate(self.h):
                # Extract this layer's config from the model config
                block.set_layer_bit_width_config(config_name)
        
        else:
            # Direct ModelBitWIdthConfig object
            # model_config = config_name_or_obj
            config_name = 'custom'

            # Update current config name
            self.current_model_bit_width_config = config_name

            for i, block in enumerate(self.h):
                layer_config = config_name_or_obj[i]
                block.set_layer_bit_width_config(layer_config)   

        

        