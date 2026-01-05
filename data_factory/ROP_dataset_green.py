"""ROPDataset with Green Channel Enhancement for hemorrhage detection.

Modification: Instead of standard RGB, apply green channel weighting:
- R: 0.9x
- G: 1.3x (boost green for better hemorrhage contrast)
- B: 0.9x

This preserves 3-channel input (keeps ImageNet pretrained weights)
while emphasizing the channel that best shows hemorrhages.
"""

import cv2
import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, Any
import albumentations as A

from data_factory.ROP_dataset import ROPDataset as BaseROPDataset


class ROPDatasetGreenEnhanced(BaseROPDataset):
    """ROPDataset with green channel enhancement."""
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        config,
        is_train: bool = True,
        transform: Optional[A.Compose] = None,
        green_weight: float = 1.3  # How much to boost green channel
    ):
        """Initialize with green channel enhancement.
        
        Args:
            dataframe: DataFrame with image and mask paths
            config: Configuration object
            is_train: Whether this is training data
            transform: Optional custom transform
            green_weight: Multiplier for green channel (default 1.3)
        """
        super().__init__(dataframe, config, is_train, transform)
        self.green_weight = green_weight
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image and apply green channel enhancement.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Green-enhanced RGB image [H, W, 3]
        """
        # Load normal RGB
        image = super()._load_image(image_path)
        
        # Apply green channel weighting
        image = self._enhance_green_channel(image)
        
        return image
    
    def _enhance_green_channel(self, image: np.ndarray) -> np.ndarray:
        """Enhance green channel for better hemorrhage contrast.
        
        Args:
            image: RGB image [H, W, 3]
            
        Returns:
            Green-enhanced RGB image [H, W, 3]
        """
        # Convert to float for multiplication
        image_float = image.astype(np.float32)
        
        # Apply channel-wise weights
        # R: reduce slightly (0.9x)
        # G: boost for hemorrhages (1.3x)
        # B: reduce slightly (0.9x)
        weights = np.array([0.9, self.green_weight, 0.9])
        
        # Apply weights
        image_enhanced = image_float * weights[None, None, :]
        
        # Clip to valid range and convert back to uint8
        image_enhanced = np.clip(image_enhanced, 0, 255).astype(np.uint8)
        
        return image_enhanced
