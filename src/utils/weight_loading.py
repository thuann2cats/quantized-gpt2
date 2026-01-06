"""
Weight loading utilities for pretrained GPT-2 models.

This module handles loading pretrained GPT-2 weights from HuggingFace
into our quantized model, with correct Conv1D → Linear transposition.
"""

import torch
from typing import Optional
from transformers import GPT2ForQuestionAnswering


def load_pretrained_gpt2_weights(
    quantized_model,
    pretrained_model_name: str = "gpt2",
    strict: bool = False,
    device: Optional[str] = None,
) -> tuple:
    """
    Load pretrained GPT-2 weights into a quantized model.
    
    This function:
    1. Loads the original GPT-2 model from HuggingFace
    2. Extracts its state_dict
    3. Creates a new state_dict with:
       - Conv1D → Linear transposition for MLP and attention
       - c_attn splitting into q_proj, k_proj, v_proj
       - Direct copying for embeddings and LayerNorm
    4. Loads the new state_dict into the quantized model

    Args:
        quantized_model: Our QuantizedGPT2ForQuestionAnswering instance
        pretrained_model_name: HuggingFace model name (default: "gpt2")
        strict: If True, raise error on missing/unexpected keys (default: False)
        device: Device to load model on (default: None, uses CPU)
    
    Returns:
        tuple: (missing_keys, unexpected_keys) from load_state_dict()    

    Note:
        - LoRA parameters will be in missing_keys (expected, they'll be trained)
        - Must call set_model_bit_width_config() before using the model        
    """
    # Load original GPT-2 model from HuggingFace
    original_model = GPT2ForQuestionAnswering.from_pretrained(
        pretrained_model_name,
        torch_dtype=torch.float32,  # Use float32 for exact comparison
    )

    if device:
        original_model = original_model.to(device)

    # Get state dict
    original_state = original_model.state_dict()

    # Create new state dict for quantized model
    new_state = {}    

    # Embeddings (no transposition)
    new_state['transformer.wte.weight'] = original_state['transformer.wte.weight']
    new_state['transformer.wpe.weight'] = original_state['transformer.wpe.weight']    

    # Final LayerNorm (no transposition)
    new_state['transformer.ln_f.weight'] = original_state['transformer.ln_f.weight']
    new_state['transformer.ln_f.bias'] = original_state['transformer.ln_f.bias']    

    # QA Head (no transposition)
    new_state['qa_outputs.weight'] = original_state['qa_outputs.weight']
    new_state['qa_outputs.bias'] = original_state['qa_outputs.bias']    

    # Transformer Blocks (transposition is required)

    num_layers = original_model.config.num_hidden_layers

    # 2. Assert architecture compatibility
    # We check the actual length of the module list to be safe
    if len(quantized_model.transformer.h) != num_layers:
        raise ValueError(
            f"Architecture mismatch: Pretrained model has {num_layers} layers, "
            f"but quantized model has {len(quantized_model.transformer.h)} layers."
        )

    for i in range(num_layers):
        prefix = f'transformer.h.{i}'

        # LayerNorms (no transposition)
        new_state[f'{prefix}.ln_1.weight'] = original_state[f'{prefix}.ln_1.weight']
        new_state[f'{prefix}.ln_1.bias'] = original_state[f'{prefix}.ln_1.bias']
        new_state[f'{prefix}.ln_2.weight'] = original_state[f'{prefix}.ln_2.weight']
        new_state[f'{prefix}.ln_2.bias'] = original_state[f'{prefix}.ln_2.bias']
        
        # MLP: c_fc needs transposition.
        # Original: Conv1D [768, 3072] -> Linear [3072, 768]
        new_state[f'{prefix}.mlp.c_fc.weight'] = original_state[f'{prefix}.mlp.c_fc.weight'].T
        new_state[f'{prefix}.mlp.c_fc.bias'] = original_state[f'{prefix}.mlp.c_fc.bias']

        # MLP: c_proj needs transposition
        new_state[f'{prefix}.mlp.c_proj.weight'] = original_state[f'{prefix}.mlp.c_proj.weight'].T
        new_state[f'{prefix}.mlp.c_proj.bias'] = original_state[f'{prefix}.mlp.c_proj.bias']        

        # Attention: c_attn -> SPLIT into q_proj, k_proj, v_proj
        c_attn_weight = original_state[f'{prefix}.attn.c_attn.weight']  # [768, 2304]
        c_attn_bias = original_state[f'{prefix}.attn.c_attn.bias']      # [2304]        

        # Split along the output dimension (dim=1 for weight, dim=0 for bias)
        q_w, k_w, v_w = c_attn_weight.chunk(3, dim=1)
        q_b, k_b, v_b = c_attn_bias.chunk(3, dim=0)

        # Transpose each part (Conv1d -> Linear)
        new_state[f'{prefix}.attn.q_proj.weight'] = q_w.T  # [768, 768] → [768, 768]
        new_state[f'{prefix}.attn.q_proj.bias'] = q_b
        new_state[f'{prefix}.attn.k_proj.weight'] = k_w.T
        new_state[f'{prefix}.attn.k_proj.bias'] = k_b
        new_state[f'{prefix}.attn.v_proj.weight'] = v_w.T
        new_state[f'{prefix}.attn.v_proj.bias'] = v_b

        # Attention: c_proj needs transposition
        new_state[f'{prefix}.attn.c_proj.weight'] = original_state[f'{prefix}.attn.c_proj.weight'].T
        new_state[f'{prefix}.attn.c_proj.bias'] = original_state[f'{prefix}.attn.c_proj.bias']        


    # Load into quantized model
    missing_keys, unexpected_keys = quantized_model.load_state_dict(
        new_state,
        strict=strict
    )
        
    return missing_keys, unexpected_keys    
