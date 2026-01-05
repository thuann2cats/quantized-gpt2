import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union

from src.quantization.quantizers import SymQuantizer, AsymQuantizer
from src.quantization.lora import LoRALayer
from src.config.layer_config import LayerBitWidthConfig
from src.config.lora_config import LoRAConfig


class QuantizedLinearWithLoRA(nn.Linear):
    """
    Linear layer with switchable quantization and LoRA adapters.
    
    Mirrors LLM-QAT's QuantizeLinear logic for quantization.
    Adds multiple LoRA adapters, selectable by current weight bit-width.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        available_layer_bit_width_configs: Dict[str, LayerBitWidthConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        symmetric: bool = True,
        act_layerwise: bool = False,
        weight_layerwise: bool = False,
    ):
        super().__init__(in_features, out_features, bias=bias)

        self.act_layerwise = act_layerwise
        self.weight_layerwise = weight_layerwise
        self.available_layer_bit_width_configs = available_layer_bit_width_configs or {}

        # Set activation quantizer based on symmetric flag
        if symmetric:
            self.act_quantizer = SymQuantizer
        else:
            self.act_quantizer = AsymQuantizer

        # Initialize current state to None to force explicit configuration
        self.current_layer_bit_width_config: Optional[str] = None
        self.current_bit_width_w: Optional[int] = None
        self.current_bit_width_a: Optional[int] = None

        # Initialize LoRA adapters
        # Key: weight bit-width (as string), Value: LoRALayer
        self.lora_set = nn.ModuleDict()

        if lora_config:
            # Create a LoRA adapter for each unique weight bit-width int he configs
            # We use weight bit-width as key because LoRA is tied to the weight precision
            unique_w_bits = set()
            for config in self.available_layer_bit_width_configs.values():
                if config.bit_width_w is not None:
                    unique_w_bits.add(config.bit_width_w)

            for w_bit in unique_w_bits:
                self.lora_set[str(w_bit)] = LoRALayer(
                    in_features,
                    out_features,
                    rank=lora_config.r,
                    lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout,
                )
    
    def set_layer_bit_width_config(self, config_name_or_obj: Union[str, LayerBitWidthConfig]):
        """
        Set the current quantization configuration.
        """
        if isinstance(config_name_or_obj, str):
            if config_name_or_obj not in self.available_layer_bit_width_configs:
                raise ValueError(f"Config '{config_name_or_obj}' not found in available configs")
            self.current_layer_bit_width_config = config_name_or_obj
            config = self.available_layer_bit_width_configs[config_name_or_obj]
        else:
            self.current_layer_bit_width_config = "custom"
            config = config_name_or_obj

        self.current_bit_width_w = config.bit_width_w
        self.current_bit_width_a = config.bit_width_a

        # Assert LoRA adapter exists if LoRA is enabled for this layer
        if len(self.lora_set) > 0:
            w_bit_str = str(self.current_bit_width_w)
            if w_bit_str not in self.lora_set:
                raise ValueError(f"No LoRA adapter found for weight bit-width {self.current_bit_width_w}")

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Ensure config is set
        if self.current_bit_width_w is None or self.current_bit_width_a is None:
            raise RuntimeError("Quantization config not set. Call set_layer_bit_width_config() first.")

        # 1. Weight quantization (mirrors LLM-QAT logic)
        real_weights = self.weight

        if self.current_bit_width_w >= 32:
            weight = self.weight
        elif self.current_bit_width_w >= 3:
            # 3-bit or higher: Use SymQuantizer
            weight_clip_val = torch.tensor([-2.0, 2.0], device=real_weights.device)
            weight = SymQuantizer.apply(
                real_weights, weight_clip_val, self.current_bit_width_w, self.weight_layerwise
            )
        else:
            # < 3 bits: Custom logic from LLM-QAT
            # < 3 bits (1 or 2 bits): Custom logic from LLM-QAT
            if self.current_bit_width_w == 1:
                if self.weight_layerwise:
                    scaling_factor = torch.mean(torch.abs(real_weights)).detach()
                else:
                    scaling_factor = torch.mean(
                        torch.abs(real_weights), dim=1, keepdim=True
                    ).detach()
                quan_weights_no_grad = scaling_factor * (
                    torch.sign(real_weights / scaling_factor)
                )
            else:
                # 2 bits
                num_bits = 2 ** (self.current_bit_width_w - 1)
                clip_val = 1 - 1e-2
                if self.weight_layerwise:
                    scaling_factor = 2 * torch.mean(torch.abs(real_weights)).detach()
                else:
                    scaling_factor = (
                        2 * torch.mean(torch.abs(real_weights), dim=1, keepdim=True).detach()
                    )
                quan_weights_no_grad = (
                    scaling_factor
                    * (
                        torch.round(
                            torch.clamp(
                                real_weights / scaling_factor, -clip_val, clip_val
                            )
                            * num_bits
                            - 0.5
                        )
                        + 0.5
                    )
                    / num_bits
                )

            # STE for low-bit weights
            weight = (
                quan_weights_no_grad.detach() - real_weights.detach() + real_weights
            )

        # 2. Activation quantization (mirrors LLM-QAT logic)
        if 2 < self.current_bit_width_a < 32:
            act_clip_val = torch.tensor([-2.0, 2.0], device=input_.device)
            input_ = self.act_quantizer.apply(
                input_, act_clip_val, self.current_bit_width_a, self.act_layerwise
            )

        # 3. Linear operation
        out = F.linear(input_, weight, self.bias)

        # 4. LoRA addition
        # Add LoRA output if an adapter exists for the current weight bit-width
        w_bit_str = str(self.current_bit_width_w)
        if w_bit_str in self.lora_set:
            out += self.lora_set[w_bit_str](input_)
        
        return out


        

