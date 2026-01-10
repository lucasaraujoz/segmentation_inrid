"""ROPDataset for Patch-Based Segmentation.

This module creates patches from full-resolution images for training.
Each patch is extracted with optional overlap using a sliding window approach.

Responsibilities:
- Extract patches from original images and masks
- Apply CLAHE preprocessing to full image before patching
- Handle edge cases with padding
- Maintain spatial information for reconstruction
"""

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ROPDatasetPatches(Dataset):
    """PyTorch Dataset for patch-based ROP segmentation."""
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        config,
        patch_size: int = 512,
        overlap: int = 0,
        is_train: bool = True,
        transform: Optional[A.Compose] = None
    ):
        """Initialize ROPDatasetPatches.
        
        Args:
            dataframe: DataFrame with image and mask paths
            config: Configuration object
            patch_size: Size of square patches (default: 512)
            overlap: Overlap between patches in pixels (default: 0)
            is_train: Whether this is training data (affects augmentation)
            transform: Optional custom transform (overrides default)
        """
        self.df = dataframe.reset_index(drop=True)
        self.config = config
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.is_train = is_train
        
        # Calculate all patch positions for all images
        self.patch_info = self._calculate_patch_positions()
        
        # Set up transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform()
    
    def _calculate_patch_positions(self) -> list:
        """Calculate patch positions for all images.
        
        Returns:
            List of dicts containing image_idx and patch coordinates
        """
        patch_list = []
        
        for img_idx in range(len(self.df)):
            # Load image to get dimensions
            img_path = self.df.iloc[img_idx]['image_path']
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            img_h, img_w = img.shape[:2]
            
            # Calculate number of patches in each dimension
            n_patches_w = int(np.ceil((img_w - self.patch_size) / self.stride)) + 1
            n_patches_h = int(np.ceil((img_h - self.patch_size) / self.stride)) + 1
            
            # Generate all patch coordinates
            for i in range(n_patches_h):
                for j in range(n_patches_w):
                    y = i * self.stride
                    x = j * self.stride
                    
                    # Adjust last patches to fit within image bounds
                    if y + self.patch_size > img_h:
                        y = img_h - self.patch_size
                    if x + self.patch_size > img_w:
                        x = img_w - self.patch_size
                    
                    patch_list.append({
                        'image_idx': img_idx,
                        'x': x,
                        'y': y,
                        'grid_i': i,
                        'grid_j': j,
                        'img_w': img_w,
                        'img_h': img_h
                    })
        
        return patch_list
    
    def _get_default_transform(self) -> A.Compose:
        """Get default augmentation transform.
        
        Returns:
            Albumentations composition
        """
        if self.is_train:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def _load_and_preprocess_image(self, img_path: str) -> np.ndarray:
        """Load image and apply CLAHE preprocessing.
        
        Args:
            img_path: Path to image file
            
        Returns:
            Preprocessed RGB image as numpy array
        """
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply CLAHE if configured
        if self.config.apply_clahe:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_grid_size
            )
            l = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            lab = cv2.merge([l, a, b])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return image
    
    def _load_masks(self, row: pd.Series, img_shape: Tuple[int, int]) -> np.ndarray:
        """Load and combine masks for all classes.
        
        Args:
            row: DataFrame row with mask paths
            img_shape: Shape of the image (H, W)
            
        Returns:
            Combined mask array of shape (num_classes, H, W)
        """
        masks = []
        
        for class_name in self.config.classes:
            col_name = f'mask_{class_name}_paths'
            
            if col_name not in row or not isinstance(row[col_name], list) or len(row[col_name]) == 0:
                # Create empty mask
                mask = np.zeros(img_shape, dtype=np.float32)
            else:
                # Load and combine all masks for this class
                combined_mask = np.zeros(img_shape, dtype=np.float32)
                
                for mask_path in row[col_name]:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        # Normalize to [0, 1]
                        mask = (mask > 0).astype(np.float32)
                        combined_mask = np.maximum(combined_mask, mask)
                
                mask = combined_mask
            
            masks.append(mask)
        
        # Stack masks: (num_classes, H, W)
        return np.stack(masks, axis=0)
    
    def __len__(self) -> int:
        """Return total number of patches."""
        return len(self.patch_info)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a patch.
        
        Args:
            idx: Patch index
            
        Returns:
            Dictionary containing patch data
        """
        patch = self.patch_info[idx]
        img_idx = patch['image_idx']
        row = self.df.iloc[img_idx]
        
        # Load full image and masks
        image = self._load_and_preprocess_image(row['image_path'])
        masks = self._load_masks(row, image.shape[:2])
        
        # Extract patch
        x, y = patch['x'], patch['y']
        patch_image = image[y:y+self.patch_size, x:x+self.patch_size].copy()
        patch_masks = masks[:, y:y+self.patch_size, x:x+self.patch_size].copy()
        
        # Apply transforms
        if self.transform:
            # Transpose masks from (C, H, W) to (H, W, C) for albumentations
            patch_masks_hwc = np.transpose(patch_masks, (1, 2, 0))
            
            transformed = self.transform(
                image=patch_image,
                mask=patch_masks_hwc
            )
            
            patch_image = transformed['image']
            patch_masks = transformed['mask']
            
            # Transpose back to (C, H, W)
            if isinstance(patch_masks, torch.Tensor):
                patch_masks = patch_masks.permute(2, 0, 1)
            else:
                patch_masks = np.transpose(patch_masks, (2, 0, 1))
                patch_masks = torch.from_numpy(patch_masks)
        
        return {
            'image': patch_image,
            'mask': patch_masks,
            'image_name': row['image_name'],
            'image_idx': img_idx,
            'patch_x': x,
            'patch_y': y,
            'grid_i': patch['grid_i'],
            'grid_j': patch['grid_j'],
            'img_width': patch['img_w'],
            'img_height': patch['img_h']
        }
