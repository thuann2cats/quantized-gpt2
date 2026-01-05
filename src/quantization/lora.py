import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer.
    
    Implements: output = (input @ A) @ B^T * scaling
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha

        self.scaling = lora_alpha / rank

        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
        
        # LoRA matrices
        # A initialized with Kaiming uniform
        # B initialized to zeros
        # This ensures LoRA initially adds zero to the base weights
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize LoRA parameters"""
        # Initialize A with kaiming uniform (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: (x @ A) @ B^T * scaling
        
        Args:
            x: Input tensor [..., in_features]
        
        Returns:
            Output tensor [..., out_features]
        """
        # Apply dropout to input
        x = self.lora_dropout(x)

        # Low-rank computation
        intermediate = x @ self.lora_A
        output = intermediate @ self.lora_B.T

        # Apply scaling
        output = output * self.scaling

        return output
    
    def extra_repr(self) -> str:
        """String representation for print(model)"""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"lora_alpha={self.lora_alpha}, "
            f"scaling={self.scaling:.4f}"
        )
    