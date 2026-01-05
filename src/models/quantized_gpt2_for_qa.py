import torch
import torch.nn as nn
from typing import Dict, Optional, Union
from transformers.models.gpt2.modeling_gpt2 import GPT2ForQuestionAnswering

from src.models.quantized_gpt2_model import QuantizedGPT2Model
from src.config.model_config import ModelBitWidthConfig
from src.config.lora_config import LoRAConfig


class QuantizedGPT2ForQuestionAnswering(GPT2ForQuestionAnswering):
    """
    Quantized version of GPT2ForQuestionAnswering.
    
    Inherits from transformers' GPT2ForQuestionAnswering and replaces
    the base GPT2Model with QuantizedGPT2Model.
    
    Keeps unchanged:
    - qa_outputs: Linear layer for start/end logits (NOT quantized)
    - forward() method (inherited)
    
    Replaces:
    - transformer: GPT2Model → QuantizedGPT2Model
    
    Adds:
    - available_model_bit_width_configs: Dict[str, ModelBitWidthConfig]
    - current_model_bit_width_config: Optional[str]
    - set_model_bit_width_config(): delegates to transformer
    """

    def __init__(
        self,
        config,
        available_model_bit_width_configs: Dict[str, ModelBitWidthConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ):
        """
        Initialize QuantizedGPT2ForQuestionAnswering.
        
        Args:
            config: GPT2Config or QuantizedGPT2Config
            available_model_bit_width_configs: Dict mapping config names to ModelBitWidthConfig
                Example: {
                    '2bit': ModelBitWidthConfig([...12 layer configs...]),
                    '4bit': ModelBitWidthConfig([...12 layer configs...]),
                }
            lora_config: LoRAConfig for all LoRA adapters
        """

        # Initialize parent class (creates transformer and qa_outputs)
        super().__init__(config)

        # Store model-level configs
        self.available_model_bit_width_configs = available_model_bit_width_configs or {}
        self.current_model_bit_width_config: Optional[str] = None

        # Delete the original transformer created by parent (This is GPT2Model, we need QuantizedGPT2Model)
        del self.transformer

        # Create quantized transformer
        self.transformer = QuantizedGPT2Model(
            config=config,
            available_model_bit_width_configs=available_model_bit_width_configs,
            lora_config=lora_config
        )

        # NOTE: self.qa_outputs already exists from super().__init__()
        # It's a nn.Linear(config.hidden_size, 2) for start/end logits
        # We keep it as-is (NOT quantized)

    def set_model_bit_width_config(
        self,
        config_name_or_obj: Union[str, ModelBitWidthConfig]
    ):
        """
        Set the bit-width configuration for the entire model.

        This delegates to the transformer, which cascades to all blocks and layers.
        
        Args:
            config_name_or_obj: Either:
                - str: name of config in available_model_bit_width_configs (e.g., '4bit')
                - ModelBitWidthConfig: custom config object (list of 12 LayerBitWidthConfigs)        
        """
        # Handle both string names and direct config objects
        if isinstance(config_name_or_obj, str):
            config_name = config_name_or_obj

            # Validate config name exists
            if config_name not in self.available_model_bit_width_configs:
                available = list(self.available_model_bit_width_configs.keys())
                raise ValueError(
                    f"Config '{config_name}' not found. "
                    f"Available configs: {available}"
                )                

        else:
            config_name = 'custom'

        # Update own state
        self.current_model_bit_width_config = config_name

        # Delete to transformer (which will cascade to all blocks)
        self.transformer.set_model_bit_width_config(config_name_or_obj)
