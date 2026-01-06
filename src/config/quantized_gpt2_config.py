# """Configuration class for QuantizedGPT2 models."""
# from typing import Dict
# from transformers import GPT2Config
# from .model_config import ModelBitWidthConfig
# from .lora_config import LoRAConfig


# class QuantizedGPT2Config(GPT2Config):
#     """Configuration for QuantizedGPT2ForQuestionAnswering."""

#     def __init__(
#         self,
#         available_model_bit_width_configs: Dict[str, ModelBitWidthConfig] = None,
#         lora_config: LoRAConfig = None,
#         **kwargs
#     ):
#         # Initialize parent GPT2Config
#         super().__init__(**kwargs)

#         # Add quantization-specific fields
#         self.available_model_bit_width_configs = available_model_bit_width_configs or {}
#         self.lora_config = lora_config or LoRAConfig()

#     @property
#     def model_type(self) -> str:
#         """Return model type for HuggingFace compatibility."""
#         return "quantized_gpt2"        