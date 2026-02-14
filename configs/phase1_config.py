"""
Configuration file for SSMD-UNet Phase 1
Central place for all hyperparameters
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Phase1Config:
    """Configuration for Phase 1 (Unsupervised Pre-training)"""
    
    # ==================== DATA ====================
    data_root: str = "/home/lucas/mestrado/tapi_inrid/eye_pacs/data/train"
    image_size: int = 512
    
    # Pre-processing
    apply_blur: bool = True
    blur_kernel: int = 5
    normalize: bool = True  # [0, 1]
    
    # ==================== MODEL ====================
    in_channels: int = 3
    out_channels: int = 3
    
    # Encoder channels (as per article)
    encoder_channels: list = None
    
    def __post_init__(self):
        if self.encoder_channels is None:
            self.encoder_channels = [64, 128, 256, 512, 1024]
    
    # ==================== TRAINING ====================
    # Article uses batch 16, but 8 fits better in 15.57GB GPU with this model size
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4  # 0.0001 as per article
    momentum: float = 0.9
    
    # Checkpoints
    checkpoint_dir: str = "checkpoints/phase1"
    checkpoint_interval: int = 10  # Save every N epochs
    vis_interval: int = 10  # Visualize every N epochs
    
    # ==================== DEVICE ====================
    device: str = "cuda"  # or "cpu"
    num_workers: int = 12  # Increased from 4 for faster data loading
    pin_memory: bool = True
    prefetch_factor: int = 2  # Prefetch 2 batches per worker
    
    # ==================== LOGGING ====================
    log_level: str = "INFO"
    seed: Optional[int] = None  # Set for reproducibility
    
    def __str__(self):
        """Pretty print configuration"""
        lines = [
            "="*80,
            "SSMD-UNet Phase 1 Configuration",
            "="*80,
            "\n[DATA]",
            f"  Data root: {self.data_root}",
            f"  Image size: {self.image_size}×{self.image_size}",
            f"  Pre-processing: Blur={self.apply_blur}, Normalize={self.normalize}",
            "\n[MODEL]",
            f"  Encoder channels: {self.encoder_channels}",
            f"  Input/Output channels: {self.in_channels}/{self.out_channels}",
            "\n[TRAINING]",
            f"  Batch size: {self.batch_size}",
            f"  Epochs: {self.num_epochs}",
            f"  Optimizer: SGD(lr={self.learning_rate}, momentum={self.momentum})",
            f"  Loss: MSE",
            "\n[CHECKPOINTS]",
            f"  Save interval: {self.checkpoint_interval} epochs",
            f"  Visualization interval: {self.vis_interval} epochs",
            f"  Directory: {self.checkpoint_dir}",
            "\n[DEVICE]",
            f"  Device: {self.device}",
            f"  Workers: {self.num_workers}",
            "="*80,
        ]
        return "\n".join(lines)


# Default configuration (as per article)
DEFAULT_CONFIG = Phase1Config()


# Quick preset configurations
QUICK_TEST_CONFIG = Phase1Config(
    batch_size=4,
    num_epochs=2,
    num_workers=2,
    checkpoint_dir="checkpoints/phase1_test",
    checkpoint_interval=1,
    vis_interval=1,
)

SMALL_CONFIG = Phase1Config(
    batch_size=8,
    num_epochs=10,
    checkpoint_dir="checkpoints/phase1_small",
)

PRODUCTION_CONFIG = Phase1Config(
    batch_size=16,
    num_epochs=100,
    checkpoint_dir="checkpoints/phase1_production",
)


if __name__ == "__main__":
    print(DEFAULT_CONFIG)
    print("\nQuick test config:")
    print(QUICK_TEST_CONFIG)
    print("\nSmall config:")
    print(SMALL_CONFIG)
    print("\nProduction config:")
    print(PRODUCTION_CONFIG)
