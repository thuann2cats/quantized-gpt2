import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Callable
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from src.quantization.quantized_linear import QuantizedLinearWithLoRA
from src.quantization.quantizers import SymQuantizer
from src.config.layer_config import LayerBitWidthConfig
from src.config.lora_config import LoRAConfig


def eager_attention_forward(module, query, key, value, attention_mask, **kwargs):
    """
    Eager attention implementation (copied from transformers for compatibility).
    This mirrors the implementation in modeling_gpt2.py.
    """    
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # Layer-wise attention scaling
    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)

    if not module.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    if attention_mask is not None:
        # Apply the attention mask
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights    


class QuantizedGPT2Attention(GPT2Attention):
    """
    Quantized version of GPT2Attention with separate QKV projections and KV cache quantization.
    
    Key differences from GPT2Attention:
    1. Replaces single c_attn Conv1D with separate q_proj, k_proj, v_proj (QuantizedLinearWithLoRA)
    2. Adds KV cache quantization using SymQuantizer from LLM-QAT
    3. Inherits _upcast_and_reordered_attn method from GPT2Attention
    
    Inherits from GPT2Attention to reuse:
    - _upcast_and_reordered_attn() method
    - Causal mask buffer registration
    - Dropout layers
    """

    def __init__(
        self,
        config,
        is_cross_attention: bool = False,
        layer_idx: Optional[int] = None,
        available_layer_bit_width_configs: Optional[Dict[str, LayerBitWidthConfig]] = None,
        lora_config: Optional[LoRAConfig] = None,
    ):
        # Call parent __init__ to set up basic attributes
        # This creates c_attn and c_proj as Conv1D, which we'll immediately override
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)

        self.available_layer_bit_width_configs = available_layer_bit_width_configs or {}
        self.current_layer_bit_width_config: Optional[str] = None

        # KV cache quantization parameters (will be set dynamically)
        self.kv_bits: Optional[int] = None
        self.act_clip_val_k = nn.Parameter(
            torch.tensor([-2.0, 2.0]), requires_grad=False
        )
        self.act_clip_val_v = nn.Parameter(
            torch.tensor([-2.0, 2.0]), requires_grad=False
        )
        self.act_quantizer_k = SymQuantizer
        self.act_quantizer_v = SymQuantizer

        # Override c_attn with separate QKV projections
        # Parent created c_attn as Conv1D(3 * embed_dim, embed_dim)
        # We split into three separate QuantizedLinearWithLoRA layers
        del self.c_attn  # Remove the Conv1D version

        if self.is_cross_attention:
            # For cross-attention: separate K, V from encoder; Q from decoder
            del self.q_attn  # Remove Conv1D version
            self.q_attn = QuantizedLinearWithLoRA(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                bias=True,
                available_layer_bit_width_configs=available_layer_bit_width_configs,
                lora_config=lora_config
            )

            # c_attn for K, V (from encoder)
            self.k_proj = QuantizedLinearWithLoRA(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                bias=True,
                available_layer_bit_width_configs=available_layer_bit_width_configs,
                lora_config=lora_config,
            )
            self.v_proj = QuantizedLinearWithLoRA(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                bias=True,
                available_layer_bit_width_configs=available_layer_bit_width_configs,
                lora_config=lora_config,
            )
        else:
            # Self-attention: Q, K, V all from same input
            self.q_proj = QuantizedLinearWithLoRA(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                bias=True,
                available_layer_bit_width_configs=available_layer_bit_width_configs,
                lora_config=lora_config,
            )
            self.k_proj = QuantizedLinearWithLoRA(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                bias=True,
                available_layer_bit_width_configs=available_layer_bit_width_configs,
                lora_config=lora_config,
            )
            self.v_proj = QuantizedLinearWithLoRA(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                bias=True,
                available_layer_bit_width_configs=available_layer_bit_width_configs,
                lora_config=lora_config,
            )

        # Override c_proj with QuantizedLinearWithLoRA
        # Parent created c_proj as Conv1D(embed_dim, embed_dim)
        del self.c_proj
        self.c_proj = QuantizedLinearWithLoRA(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=True,
            available_layer_bit_width_configs=available_layer_bit_width_configs,
            lora_config=lora_config,
        )

    def set_layer_bit_width_config(self, config_name_or_obj: Union[str, LayerBitWidthConfig]):
        """
        Set the current quantization configuration and propagate to all sub-layers.
        """
        if isinstance(config_name_or_obj, str):
            if config_name_or_obj not in self.available_layer_bit_width_configs:
                raise ValueError(
                    f"Config '{config_name_or_obj}' not found in available configs. "
                    f"Available: {list(self.available_layer_bit_width_configs.keys())}"
                )
            self.current_layer_bit_width_config = config_name_or_obj
            config = self.available_layer_bit_width_configs[config_name_or_obj]
        else:
            self.current_layer_bit_width_config = "custom"
            config = config_name_or_obj

        # Set KV cache quantization bit-width
        self.kv_bits = config.bit_width_kv

        # Propagate to all linear layers
        if self.is_cross_attention:
            self.q_attn.set_layer_bit_width_config(config_name_or_obj)
        else:
            self.q_proj.set_layer_bit_width_config(config_name_or_obj)

        self.k_proj.set_layer_bit_width_config(config_name_or_obj)
        self.v_proj.set_layer_bit_width_config(config_name_or_obj)
        self.c_proj.set_layer_bit_width_config(config_name_or_obj)

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ):
        """
        Forward pass mirroring GPT2Attention with quantized QKV projections and KV cache quantization."""
        # Safety check: ensure config is set
        if self.kv_bits is None:
            raise RuntimeError(
                "kv_bits not set! Call set_layer_bit_width_config() before forward pass"
            )

        # Handle cross-attention logic (mirroring GPT2Attention)
        is_cross_attention = encoder_hidden_states is not None

        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_value from cache
                    curr_past_key_values = past_key_values.cross_attention_cache
                else:
                    curr_past_key_values = past_key_values.self_attention_cache
            else:
                curr_past_key_values = past_key_values

        # QKV Projection (CHANGE: separate projections instead of split)
        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `QuantizedGPT2Attention(..., is_cross_attention=True)`."
                )
            query_states = self.q_attn(hidden_states)
            attention_mask = encoder_attention_mask

            # Try to get key/value states from cache if possible
            if past_key_values is not None and is_updated:
                key_states = curr_past_key_values.layers[self.layer_idx].keys
                value_states = curr_past_key_values.layers[self.layer_idx].values
            else:
                # Project K, V from encoder hidden states
                key_states = self.k_proj(encoder_hidden_states)
                value_states = self.v_proj(encoder_hidden_states)

                # KV cache quantization (CHANGE: quantize BEFORE reshaping)
                if self.kv_bits < 32:
                    key_states = self.act_quantizer_k.apply(
                        key_states, self.act_clip_val_k, self.kv_bits, False
                    )
                    value_states = self.act_quantizer_v.apply(
                        value_states, self.act_clip_val_v, self.kv_bits, False
                    )
                
                # Reshape to multi-head format
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        
        else:
            # Self-attention: Q, K, V from same input (CHANGE: separate projections)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            # KV cache quantization (CHANGE: quantize BEFORE reshaping)
            if self.kv_bits < 32:
                key_states = self.act_quantizer_k.apply(
                    key_states, self.act_clip_val_k, self.kv_bits, False
                )
                value_states = self.act_quantizer_v.apply(
                    value_states, self.act_clip_val_v, self.kv_bits, False
                )
            
            # Reshape to multi-head format
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)
        
        # Reshape query
        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        # KV cache update (mirroring GPT2Attention)
        if (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            # save all key/value_layer to cache to be re-used for fast auto-regressive generation
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_values.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )
            # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
            if is_cross_attention:
                past_key_values.is_updated[self.layer_idx] = True

        # Attention computation (mirroring GPT2Attention's logic)
        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation not in [None, "eager"]:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if using_eager and self.reorder_and_upcast_attn:
            # Use inherited _upcast_and_reordered_attn method from GPT2Attention
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                **kwargs,
            )

        # Reshape back and apply output projection
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights