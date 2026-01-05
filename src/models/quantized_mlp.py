import torch
import torch.nn as nn
from typing import Dict, Optional, Union
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

from src.quantization.quantized_linear import QuantizedLinearWithLoRA
from src.config.layer_config import LayerBitWidthConfig
from src.config.lora_config import LoRAConfig


class QuantizedGPT2MLP(GPT2MLP):
    """
    Quantized version of GPT2MLP.
    Replaces Conv1D layers with QuantizedLinearWithLoRA.
    """

    def __init__(
        self,
        intermediate_size: int,
        config,
        available_layer_bit_width_configs: Dict[str, LayerBitWidthConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ):
        # We'll overwrite Conv1D that GPT2MLP.__init__ creates
        super().__init__(intermediate_size, config)

        self.available_layer_bit_width_configs = available_layer_bit_width_configs
        self.current_layer_bit_width_config = None

        embed_dim = config.hidden_size

        # Replace c_fc (Conv1D) with QuantizedLinearWithLoRA
        # Conv1D(intermediate_size, embed_dim) -> Linear(embed_dim, intermediate_size)
        self.c_fc = QuantizedLinearWithLoRA(
            in_features=embed_dim,
            out_features=intermediate_size,
            available_layer_bit_width_configs=available_layer_bit_width_configs,
            lora_config=lora_config
        )

        # Replace c_proj (Conv1D) with QuantizedLinearWithLoRA
        # Conv1D(embed_dim, intermediate_size) -> Linear(intermediate_size, embed_dim)
        self.c_proj = QuantizedLinearWithLoRA(
            in_features=intermediate_size,
            out_features=embed_dim,
            available_layer_bit_width_configs=available_layer_bit_width_configs,
            lora_config=lora_config
        )

    def set_layer_bit_width_config(self, config_name_or_obj: Union[str, LayerBitWidthConfig]):
        """
        Propagate config to child linear layers.
        """
        self.current_layer_bit_width_config = config_name_or_obj
        self.c_fc.set_layer_bit_width_config(config_name_or_obj)
        self.c_proj.set_layer_bit_width_config(config_name_or_obj)

    # forward() is inherited from GPT2MLP and works unchanged because QuantizedLinearWithLoRA is callable like Conv1D/Linear
