"""
Configuration for SSMD-UNet Phase 1 optimized for 24GB GPU
Maximizes speed and model capacity
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Phase1Config24GB:
    """Configuration for Phase 1 on 24GB GPU"""
    
    # ==================== DATA ====================
    data_root: str = "/home/lucas/mestrado/tapi_inrid/eye_pacs/data/train"
    image_size: int = 512
    
    # Pre-processing
    apply_blur: bool = True
    blur_kernel: int = 5
    normalize: bool = True
    
    # ==================== MODEL ====================
    in_channels: int = 3
    out_channels: int = 3
    
    # FULL model for 24GB GPU
    use_full_model: bool = True  # [64, 128, 256, 512, 1024] instead of [32, 64, 128, 256, 512]
    
    # ==================== TRAINING ====================
    batch_size: int = 24  # MUITO maior com 24GB! (antes era 8)
    num_epochs: int = 100
    learning_rate: float = 1e-4
    momentum: float = 0.9
    
    # Checkpoints
    checkpoint_dir: str = "checkpoints/phase1_24gb"
    checkpoint_interval: int = 10
    vis_interval: int = 10
    
    # ==================== DEVICE ====================
    device: str = "cuda"
    num_workers: int = 16  # Mais workers para GPU potente
    pin_memory: bool = True
    prefetch_factor: int = 4  # Mais prefetch
    
    # ==================== LOGGING ====================
    log_level: str = "INFO"
    seed: Optional[int] = None
    
    def __str__(self):
        lines = [
            "="*80,
            "SSMD-UNet Phase 1 Configuration - 24GB GPU OPTIMIZED",
            "="*80,
            "\n[DATA]",
            f"  Data root: {self.data_root}",
            f"  Image size: {self.image_size}×{self.image_size}",
            "\n[MODEL]",
            f"  Model: {'FULL [64,128,256,512,1024]' if self.use_full_model else 'OPTIMIZED [32,64,128,256,512]'}",
            f"  Parameters: {'~31M' if self.use_full_model else '~8M'}",
            "\n[TRAINING - 24GB GPU]",
            f"  Batch size: {self.batch_size} (3x larger!)",
            f"  Epochs: {self.num_epochs}",
            f"  Optimizer: SGD(lr={self.learning_rate}, momentum={self.momentum})",
            f"  Workers: {self.num_workers}",
            "\n[SPEED ESTIMATE]",
            f"  Expected time/epoch: ~10-15 min (with batch 24)",
            f"  Total time (100 epochs): ~20-25 hours",
            "="*80,
        ]
        return "\n".join(lines)
