"""Configuration dataclasses for all kernel implementations."""

from dataclasses import dataclass


@dataclass
class RoPEConfig:
    """Configuration for RoPE kernel autotuning."""
    tile_qh: int  # Tile size for query heads
    tile_kh: int  # Tile size for key heads
    tile_hd: int  # Tile size for head dimension
    occupancy: int
    
    def __hash__(self):
        return hash((self.tile_qh, self.tile_kh, self.tile_hd, self.occupancy))


@dataclass
class SwiGLUConfig:
    """Configuration for SwiGLU kernel autotuning."""
    tile_size: int
    occupancy: int
    
    def __hash__(self):
        return hash((self.tile_size, self.occupancy))


@dataclass
class GEGLUConfig:
    """Configuration for GEGLU kernel."""
    block_size: int
    occupancy: int
    use_float32: bool = True
    
    def __hash__(self):
        return hash((self.block_size, self.occupancy, self.use_float32))


@dataclass
class RMSNormConfig:
    """Configuration for RMSNorm kernel autotuning."""
    use_static_persistent: bool
    tile_size_m: int  # For persistent mode
    tile_size_n: int  # Tile size for columns
    
    def __hash__(self):
        return hash((self.use_static_persistent, self.tile_size_m, self.tile_size_n))
