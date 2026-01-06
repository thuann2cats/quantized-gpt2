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

    num_layers = 12

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
# """
# Weight loading utilities for pretrained GPT-2 models.

# This module handles loading pretrained GPT-2 weights from HuggingFace
# into our quantized model, with correct Conv1D → Linear transposition.
# """

# import torch
# from typing import Optional
# from transformers import GPT2ForQuestionAnswering


# def load_pretrained_gpt2_weights(
#     quantized_model,  # QuantizedGPT2ForQuestionAnswering
#     pretrained_model_name: str = "gpt2",
#     strict: bool = False,
#     device: Optional[str] = None,
# ) -> tuple:
#     """
#     Load pretrained GPT-2 weights into a quantized model.
    
#     This function:
#     1. Loads the original GPT-2 model from HuggingFace
#     2. Extracts its state_dict
#     3. Creates a new state_dict with:
#        - Conv1D → Linear transposition for MLP and attention
#        - c_attn splitting into q_proj, k_proj, v_proj
#        - Direct copying for embeddings and LayerNorm
#     4. Loads the new state_dict into the quantized model
    
#     Args:
#         quantized_model: Our QuantizedGPT2ForQuestionAnswering instance
#         pretrained_model_name: HuggingFace model name (default: "gpt2")
#         strict: If True, raise error on missing/unexpected keys (default: False)
#         device: Device to load model on (default: None, uses CPU)
    
#     Returns:
#         tuple: (missing_keys, unexpected_keys) from load_state_dict()
        
#     Example:
#         >>> from src.models.quantized_gpt2_for_qa import QuantizedGPT2ForQuestionAnswering
#         >>> config = create_config_with_32bit()
#         >>> model = QuantizedGPT2ForQuestionAnswering(config)
#         >>> missing, unexpected = load_pretrained_gpt2_weights(model, "gpt2")
#         >>> # missing will contain LoRA parameters (expected)
#         >>> model.set_model_bit_width_config("32bit")
#         >>> # Now ready to use!
    
#     Note:
#         - LoRA parameters will be in missing_keys (expected, they'll be trained)
#         - Must call set_model_bit_width_config() before using the model
#         - For verification, use 32-bit config to match original GPT-2 exactly
#     """
#     print(f"Loading pretrained weights from '{pretrained_model_name}'...")
    
#     # Load original GPT-2 model from HuggingFace
#     print("  1. Loading original GPT-2 from HuggingFace...")
#     original_model = GPT2ForQuestionAnswering.from_pretrained(
#         pretrained_model_name,
#         torch_dtype=torch.float32,  # Use float32 for exact comparison
#     )
    
#     if device:
#         original_model = original_model.to(device)
    
#     # Get state dict
#     original_state = original_model.state_dict()
#     print(f"     ✓ Loaded {len(original_state)} keys from original model")
    
#     # Create new state dict for quantized model
#     print("  2. Creating new state_dict with transposition...")
#     new_state = {}
    
#     # ========================================
#     # SECTION 1: Embeddings (no transposition)
#     # ========================================
#     new_state['transformer.wte.weight'] = original_state['transformer.wte.weight']
#     new_state['transformer.wpe.weight'] = original_state['transformer.wpe.weight']
    
#     # ========================================
#     # SECTION 2: Final LayerNorm (no transposition)
#     # ========================================
#     new_state['transformer.ln_f.weight'] = original_state['transformer.ln_f.weight']
#     new_state['transformer.ln_f.bias'] = original_state['transformer.ln_f.bias']
    
#     # ========================================
#     # SECTION 3: QA Head (no transposition - already nn.Linear)
#     # ========================================
#     new_state['qa_outputs.weight'] = original_state['qa_outputs.weight']
#     new_state['qa_outputs.bias'] = original_state['qa_outputs.bias']
    
#     # ========================================
#     # SECTION 4: Transformer Blocks (transposition required!)
#     # ========================================
#     num_layers = 12  # GPT-2 has 12 layers
    
#     for i in range(num_layers):
#         prefix = f'transformer.h.{i}'
        
#         # ----------------------------------------
#         # LayerNorms (no transposition)
#         # ----------------------------------------
#         new_state[f'{prefix}.ln_1.weight'] = original_state[f'{prefix}.ln_1.weight']
#         new_state[f'{prefix}.ln_1.bias'] = original_state[f'{prefix}.ln_1.bias']
#         new_state[f'{prefix}.ln_2.weight'] = original_state[f'{prefix}.ln_2.weight']
#         new_state[f'{prefix}.ln_2.bias'] = original_state[f'{prefix}.ln_2.bias']
        
#         # ----------------------------------------
#         # MLP: c_fc (TRANSPOSE!)
#         # Original: Conv1D [768, 3072]
#         # Target: Linear [3072, 768]
#         # ----------------------------------------
#         new_state[f'{prefix}.mlp.c_fc.weight'] = original_state[f'{prefix}.mlp.c_fc.weight'].T
#         new_state[f'{prefix}.mlp.c_fc.bias'] = original_state[f'{prefix}.mlp.c_fc.bias']
        
#         # ----------------------------------------
#         # MLP: c_proj (TRANSPOSE!)
#         # Original: Conv1D [3072, 768]
#         # Target: Linear [768, 3072]
#         # ----------------------------------------
#         new_state[f'{prefix}.mlp.c_proj.weight'] = original_state[f'{prefix}.mlp.c_proj.weight'].T
#         new_state[f'{prefix}.mlp.c_proj.bias'] = original_state[f'{prefix}.mlp.c_proj.bias']
        
#         # ----------------------------------------
#         # Attention: c_attn → SPLIT into q_proj, k_proj, v_proj
#         # Original: Conv1D [768, 2304] (2304 = 3 * 768)
#         # Target: 3 separate Linear layers, each [768, 768]
#         # ----------------------------------------
#         c_attn_weight = original_state[f'{prefix}.attn.c_attn.weight']  # [768, 2304]
#         c_attn_bias = original_state[f'{prefix}.attn.c_attn.bias']      # [2304]
        
#         # Split along the output dimension (dim=1 for weight, dim=0 for bias)
#         # chunk(3, dim=1) splits [768, 2304] into 3 × [768, 768]
#         q_w, k_w, v_w = c_attn_weight.chunk(3, dim=1)
#         q_b, k_b, v_b = c_attn_bias.chunk(3, dim=0)
        
#         # Transpose each part (Conv1D → Linear)
#         new_state[f'{prefix}.attn.q_proj.weight'] = q_w.T  # [768, 768] → [768, 768]
#         new_state[f'{prefix}.attn.q_proj.bias'] = q_b
#         new_state[f'{prefix}.attn.k_proj.weight'] = k_w.T
#         new_state[f'{prefix}.attn.k_proj.bias'] = k_b
#         new_state[f'{prefix}.attn.v_proj.weight'] = v_w.T
#         new_state[f'{prefix}.attn.v_proj.bias'] = v_b
        
#         # ----------------------------------------
#         # Attention: c_proj (TRANSPOSE!)
#         # Original: Conv1D [768, 768]
#         # Target: Linear [768, 768]
#         # ----------------------------------------
#         new_state[f'{prefix}.attn.c_proj.weight'] = original_state[f'{prefix}.attn.c_proj.weight'].T
#         new_state[f'{prefix}.attn.c_proj.bias'] = original_state[f'{prefix}.attn.c_proj.bias']
    
#     print(f"     ✓ Created {len(new_state)} keys for quantized model")
    
#     # ========================================
#     # SECTION 5: Load into quantized model
#     # ========================================
#     print("  3. Loading weights into quantized model...")
#     missing_keys, unexpected_keys = quantized_model.load_state_dict(
#         new_state,
#         strict=strict
#     )
    
#     # ========================================
#     # SECTION 6: Report results
#     # ========================================
#     print(f"     ✓ Load complete")
    
#     # Filter LoRA keys from missing (expected)
#     lora_missing = [k for k in missing_keys if 'lora' in k.lower()]
#     non_lora_missing = [k for k in missing_keys if 'lora' not in k.lower()]
    
#     if lora_missing:
#         print(f"     ✓ {len(lora_missing)} LoRA parameters not loaded (expected, will be trained)")
    
#     if non_lora_missing:
#         print(f"     ⚠ Warning: {len(non_lora_missing)} non-LoRA parameters missing:")
#         for key in non_lora_missing[:5]:  # Show first 5
#             print(f"       - {key}")
#         if len(non_lora_missing) > 5:
#             print(f"       ... and {len(non_lora_missing) - 5} more")
    
#     if unexpected_keys:
#         print(f"     ⚠ Warning: {len(unexpected_keys)} unexpected keys:")
#         for key in unexpected_keys[:5]:  # Show first 5
#             print(f"       - {key}")
#         if len(unexpected_keys) > 5:
#             print(f"       ... and {len(unexpected_keys) - 5} more")
    
#     print("  ✅ Weight loading complete!")
#     print()
#     print("  Next steps:")
#     print("    1. Call model.set_model_bit_width_config('32bit') for verification")
#     print("    2. Run forward pass and compare outputs with original model")
    
#     return missing_keys, unexpected_keys