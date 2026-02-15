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


@dataclass
class MoEGemmConfig:
    """Configuration for MoE GEMM kernel autotuning."""
    tile_m: int   # Tile size for token dimension (also block_size for alignment)
    tile_n: int   # Tile size for output dimension
    tile_k: int   # Tile size for reduction dimension
    group_m: int  # Group size for M-block swizzling

    def __hash__(self):
        return hash((self.tile_m, self.tile_n, self.tile_k, self.group_m))
