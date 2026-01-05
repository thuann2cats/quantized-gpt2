"""Model-level bit-width configuration."""
from dataclasses import dataclass
from typing import List
from .layer_config import LayerBitWidthConfig


@dataclass
class ModelBitWidthConfig:
    """
    Bit-width configuration for entire model (all 12 layers).
    """
    layer_configs: List[LayerBitWidthConfig]
    
    def __getitem__(self, idx: int) -> LayerBitWidthConfig:
        """Allow indexing: config[0] returns layer 0's config."""
        return self.layer_configs[idx]
    
    def __len__(self) -> int:
        """Return number of layers."""
        return len(self.layer_configs)