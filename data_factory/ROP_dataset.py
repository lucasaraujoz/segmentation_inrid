"""ROPDataset: PyTorch Dataset for ROP segmentation.

This module is responsible for:
- Loading images (.jpg) and masks (.tiff)
- Applying CLAHE preprocessing to images
- Ensuring spatial alignment between images and masks
- Applying data augmentation (training vs testing)
- Converting data to PyTorch tensors

NEVER manages splits or creates DataFrames.
"""

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ROPDataset(Dataset):
    """PyTorch Dataset for ROP segmentation."""
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        config,
        is_train: bool = True,
        transform: Optional[A.Compose] = None
    ):
        """Initialize ROPDataset.
        
        Args:
            dataframe: DataFrame with image and mask paths
            config: Configuration object
            is_train: Whether this is training data (affects augmentation)
            transform: Optional custom transform (overrides default)
        """
        self.df = dataframe.reset_index(drop=True)
        self.config = config
        self.is_train = is_train
        
        # Set up transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform()
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - 'image': Preprocessed image tensor [3, H, W]
                - 'mask': Multi-label mask tensor [num_classes, H, W]
                - 'image_name': Name of the image
        """
        row = self.df.iloc[idx]
        
        # Load image
        image = self._load_image(row["image_path"])
        
        # Apply CLAHE preprocessing
        if self.config.apply_clahe:
            image = self._apply_clahe(image)
        
        # Load masks for all classes
        mask = self._load_masks(row)
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        # Convert mask to tensor if not already and permute to [C, H, W]
        if not isinstance(mask, torch.Tensor):
            # mask shape: [H, W, num_classes] -> [num_classes, H, W]
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        else:
            # If it's already a tensor, check if we need to permute
            if mask.ndim == 3 and mask.shape[2] == self.config.num_classes:
                # [H, W, num_classes] -> [num_classes, H, W]
                mask = mask.permute(2, 0, 1).float()
        
        return {
            "image": image,
            "mask": mask,
            "image_name": row["image_name"]
        }
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array in RGB format [H, W, 3]
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        CLAHE is applied to the L channel in LAB color space,
        then converted back to RGB.
        
        Args:
            image: RGB image [H, W, 3]
            
        Returns:
            CLAHE-enhanced RGB image [H, W, 3]
        """
        # Convert RGB to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size
        )
        l_clahe = clahe.apply(l)
        
        # Merge channels
        lab_clahe = cv2.merge([l_clahe, a, b])
        
        # Convert back to RGB
        image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        return image_clahe
    
    def _load_masks(self, row: pd.Series) -> np.ndarray:
        """Load and combine masks for all classes.
        
        Args:
            row: DataFrame row containing mask paths
            
        Returns:
            Multi-label mask array [H, W, num_classes]
        """
        # Initialize empty mask
        # We'll resize after loading first mask
        mask = None
        masks_list = []
        
        for class_idx, class_name in enumerate(self.config.classes):
            mask_paths_col = f"mask_{class_name}_paths"
            mask_paths = row[mask_paths_col]
            
            if isinstance(mask_paths, list) and len(mask_paths) > 0:
                # Load and combine masks for this class
                class_mask = self._load_and_combine_masks(mask_paths)
            else:
                # No mask for this class - create empty mask
                # Use size from first loaded mask or default size
                if mask is None:
                    # Use default size for now, will resize later
                    class_mask = None
                else:
                    class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
            
            if class_mask is not None:
                if mask is None:
                    # First mask loaded
                    mask = np.zeros(
                        (class_mask.shape[0], class_mask.shape[1], 
                         self.config.num_classes),
                        dtype=np.uint8
                    )
                masks_list.append(class_mask)
            else:
                masks_list.append(None)
        
        # If no masks were loaded at all, create empty mask with default size
        if mask is None:
            h, w = self.config.image_size
            mask = np.zeros((h, w, self.config.num_classes), dtype=np.uint8)
        else:
            # Fill in the mask channels
            for class_idx, class_mask in enumerate(masks_list):
                if class_mask is not None:
                    # Resize if needed to match first mask
                    if class_mask.shape != mask.shape[:2]:
                        class_mask = cv2.resize(
                            class_mask, 
                            (mask.shape[1], mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                    mask[:, :, class_idx] = class_mask
        
        return mask
    
    def _load_and_combine_masks(self, mask_paths: list) -> np.ndarray:
        """Load and combine multiple masks for a single class.
        
        For exudates, we might have both hard and soft exudate masks.
        These are combined into a single binary mask.
        
        Args:
            mask_paths: List of paths to mask files
            
        Returns:
            Combined binary mask [H, W]
        """
        combined_mask = None
        
        for mask_path in mask_paths:
            # Load mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"Warning: Failed to load mask: {mask_path}")
                continue
            
            # Binarize mask (any non-zero value becomes 1)
            # Using threshold of 0 to handle any positive values
            _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
            
            if combined_mask is None:
                combined_mask = mask
            else:
                # Ensure same size
                if mask.shape != combined_mask.shape:
                    mask = cv2.resize(
                        mask, 
                        (combined_mask.shape[1], combined_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                # Combine with OR operation
                combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Convert to binary (0 or 1)
        if combined_mask is not None:
            combined_mask = (combined_mask > 0).astype(np.uint8)
        
        return combined_mask
    
    def _get_default_transform(self) -> A.Compose:
        """Get default augmentation pipeline.
        
        Returns:
            Albumentations Compose object
        """
        if self.is_train:
            # Training augmentations
            transform = A.Compose([
                A.Resize(
                    height=self.config.image_size[0],
                    width=self.config.image_size[1]
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=45,
                    p=0.5
                ),
                A.OneOf([
                    A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05),
                    A.GridDistortion(p=1),
                    A.OpticalDistortion(distort_limit=1, p=1),
                ], p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            # Validation/test augmentations (no random ops)
            transform = A.Compose([
                A.Resize(
                    height=self.config.image_size[0],
                    width=self.config.image_size[1]
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        
        return transform
