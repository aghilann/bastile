"""Configuration dataclasses for kernel implementations."""

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
