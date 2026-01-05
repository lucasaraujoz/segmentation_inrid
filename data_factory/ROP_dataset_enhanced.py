"""
Enhanced ROP Dataset with advanced image processing.

Uses:
- Green channel enhancement
- Vessel enhancement (Frangi filter)
- Improved preprocessing pipeline
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.image_processing import preprocess_retinal_image


class ROPDatasetEnhanced(Dataset):
    """ROP Dataset with enhanced image processing."""
    
    def __init__(
        self,
        dataframe,
        config,
        is_train: bool = True,
        use_vessel_enhancement: bool = False  # DISABLED: Frangi filter is too slow for training
    ):
        """Initialize enhanced dataset.
        
        Args:
            dataframe: Pandas DataFrame with image paths
            config: Configuration object
            is_train: Training or validation mode
            use_vessel_enhancement: Whether to use Frangi vessel enhancement (WARNING: very slow!)
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.config = config
        self.is_train = is_train
        self.use_vessel_enhancement = use_vessel_enhancement
        
        # Set up transforms (same as baseline)
        if is_train:
            self.transform = self._get_training_transform()
        else:
            self.transform = self._get_validation_transform()
    
    def _get_training_transform(self):
        """Get training augmentation pipeline."""
        if isinstance(self.config.image_size, tuple):
            h, w = self.config.image_size
        else:
            h = w = self.config.image_size
            
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.OneOf([
                A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(p=1),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
            ], p=0.3),
            ToTensorV2()
        ])
    
    def _get_validation_transform(self):
        """Get validation augmentation pipeline."""
        if isinstance(self.config.image_size, tuple):
            h, w = self.config.image_size
        else:
            h = w = self.config.image_size
            
        return A.Compose([
            ToTensorV2()
        ])
    
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item with enhanced preprocessing."""
        row = self.dataframe.iloc[idx]
        
        # Load image
        image_path = row['image_path']
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize FIRST to avoid slow preprocessing on 4K images
        if isinstance(self.config.image_size, tuple):
            h, w = self.config.image_size
        else:
            h = w = self.config.image_size
        image = cv2.resize(image, (w, h))
        
        # Enhanced preprocessing: green channel + vessel enhancement
        # Returns float32 [0, 1] - convert to uint8 [0, 255] for Albumentations
        image = preprocess_retinal_image(image, use_vessel_enhancement=self.use_vessel_enhancement)
        image = (image * 255).astype(np.uint8)  # Convert back to uint8 for Albumentations
        
        # Load masks
        masks = []
        for mask_col in ['mask_microaneurysms_path', 'mask_haemorrhages_path']:
            if mask_col in row and str(row[mask_col]) != 'nan':
                mask_path = Path(row[mask_col])
                if mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    # Resize mask to match image size
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    masks.append(mask)
                else:
                    masks.append(np.zeros((h, w), dtype=np.uint8))
            else:
                masks.append(np.zeros((h, w), dtype=np.uint8))
        
        # Stack masks: [H, W, 2]
        mask = np.stack(masks, axis=-1).astype(np.float32) / 255.0
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Ensure correct format
        # Image should be [3, H, W] from ToTensorV2
        # Mask needs to be [2, H, W]
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        elif mask.shape[-1] == 2:  # [H, W, 2]
            mask = mask.permute(2, 0, 1).float()
        
        # Ensure image is float in [0, 1]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        
        # Ensure contiguous
        image = image.contiguous()
        mask = mask.contiguous()
        
        return {
            'image': image,
            'mask': mask,
            'image_name': row['image_name']
        }

