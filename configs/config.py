"""Configuration class for the ROP segmentation project.

This module contains all hyperparameters, dataset paths, and training configurations.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Config:
    """Configuration for ROP segmentation training and evaluation."""
    
    # Dataset paths
    dataset_root: str = "/home/lucas/mestrado/tapi_inrid/A. Segmentation"
    original_images_train: str = field(init=False)
    original_images_test: str = field(init=False)
    segmentation_gt_train: str = field(init=False)
    segmentation_gt_test: str = field(init=False)
    
    # Classes to segment (only exudates and haemorrhages)
    classes: List[str] = field(default_factory=lambda: [
        "exudates",  # Combining Hard and Soft Exudates
        "haemorrhages"
    ])
    
    # Mapping from folder names to class names
    folder_to_class: Dict[str, str] = field(default_factory=lambda: {
        "3. Hard Exudates": "exudates",
        "4. Soft Exudates": "exudates",
        "2. Haemorrhages": "haemorrhages"
    })
    
    # Image preprocessing
    image_size: tuple = (512, 512)
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple = (8, 8)
    
    # Multi-label segmentation (not multi-class)
    multi_label: bool = True
    num_classes: int = 2  # exudates and haemorrhages
    
    # Training hyperparameters
    batch_size: int = 8  # Maximize GPU usage (16GB available)
    num_epochs: int = 50  # Full training
    learning_rate: float = 1e-3  # Higher for OneCycleLR
    max_learning_rate: float = 1e-3  # Peak LR for OneCycleLR
    weight_decay: float = 1e-5
    
    # Gradient accumulation (simulate larger batch)
    accumulation_steps: int = 2  # Effective batch_size = 8 * 2 = 16
    
    # Loss function configuration
    loss_type: str = "dice_focal"  # Options: "bce", "dice", "focal", "dice_focal"
    dice_weight: float = 0.5
    focal_weight: float = 0.5
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Class weighting (give more importance to harder class)
    use_class_weights: bool = True
    class_weights: List[float] = field(default_factory=lambda: [
        1.0,  # exudates (baseline)
        1.3   # haemorrhages (harder, gets more weight)
    ])
    
    # Test Time Augmentation
    use_tta: bool = True
    tta_transforms: List[str] = field(default_factory=lambda: [
        'original',
        'horizontal_flip',
        'vertical_flip',
        'rotate_90',
        'rotate_180',
        'rotate_270'
    ])
    
    # Ensemble evaluation
    use_ensemble: bool = True
    ensemble_method: str = "average"  # Options: "average", "voting"
    
    # Learning rate scheduler
    scheduler_type: str = "onecycle"  # Options: "plateau", "onecycle"
    
    # Early stopping
    early_stopping_patience: int = 20  # More patience for EfficientNet-B4
    
    # Cross-validation
    n_folds: int = 5
    random_state: int = 42
    
    # Model
    model_name: str = "unet"
    encoder_name: str = "efficientnet-b4"  # Much more powerful (19M params)
    encoder_weights: str = "imagenet"
    
    # Training device
    device: str = "cuda"
    num_workers: int = 4
    
    # Paths for saving results
    output_dir: str = "outputs"
    checkpoint_dir: str = field(init=False)
    logs_dir: str = field(init=False)
    
    def __post_init__(self):
        """Initialize derived paths."""
        # Dataset paths
        self.original_images_train = os.path.join(
            self.dataset_root, "1. Original Images", "a. Training Set"
        )
        self.original_images_test = os.path.join(
            self.dataset_root, "1. Original Images", "b. Testing Set"
        )
        self.segmentation_gt_train = os.path.join(
            self.dataset_root, "2. All Segmentation Groundtruths", "a. Training Set"
        )
        self.segmentation_gt_test = os.path.join(
            self.dataset_root, "2. All Segmentation Groundtruths", "b. Testing Set"
        )
        
        # Output paths
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        
        # Create output directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def get_class_folders(self, class_name: str) -> List[str]:
        """Get folder names for a given class.
        
        Args:
            class_name: Name of the class (e.g., 'exudates', 'haemorrhages')
            
        Returns:
            List of folder names that correspond to this class
        """
        return [
            folder for folder, cls in self.folder_to_class.items()
            if cls == class_name
        ]
