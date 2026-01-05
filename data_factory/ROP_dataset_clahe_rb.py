"""
Dataset with CLAHE applied ONLY to Red and Blue channels.
Preserves Green channel contrast for hemorrhage detection.
"""

import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ROPDatasetClaheRBOnly(Dataset):
    """
    CLAHE apenas em R e B, Green preservado.
    Hipótese: CLAHE no verde estava removendo contraste de hemorragias.
    """
    
    def __init__(self, dataframe, config, is_train=True):
        self.df = dataframe.reset_index(drop=True)
        self.config = config
        self.is_train = is_train
        
        # Augmentations
        if is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.df)
    
    def _apply_clahe_rb_only(self, image):
        """
        Aplica CLAHE APENAS nos canais R e B.
        Canal G é preservado intacto.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Split channels
        r, g, b = cv2.split(image)
        
        # CLAHE apenas em R e B
        r_clahe = clahe.apply(r)
        b_clahe = clahe.apply(b)
        
        # Merge: R_clahe, G_original, B_clahe
        return cv2.merge([r_clahe, g, b_clahe])
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        image = cv2.imread(row['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.config.img_size, self.config.img_size))
        
        # CLAHE apenas em R e B
        image = self._apply_clahe_rb_only(image)
        
        # Load masks
        mask_exudates = self._load_mask(row.get('mask_exudates_paths', []))
        mask_hemorrhages = self._load_mask(row.get('mask_haemorrhages_paths', []))
        
        # Stack masks
        mask = np.stack([mask_exudates, mask_hemorrhages], axis=-1)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'].permute(2, 0, 1).float()
        
        return {
            'image': image,
            'mask': mask
        }
    
    def _load_mask(self, mask_paths):
        """Load and combine multiple masks into one binary mask."""
        if not mask_paths or (isinstance(mask_paths, list) and len(mask_paths) == 0):
            return np.zeros((self.config.img_size, self.config.img_size), dtype=np.float32)
        
        # Handle single path or list of paths
        if isinstance(mask_paths, str):
            mask_paths = [mask_paths]
        
        # Combine all masks
        combined_mask = np.zeros((self.config.img_size, self.config.img_size), dtype=np.float32)
        
        for mask_path in mask_paths:
            if pd.isna(mask_path) or not os.path.exists(mask_path):
                continue
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.config.img_size, self.config.img_size))
            combined_mask = np.maximum(combined_mask, (mask > 0).astype(np.float32))
        
        return combined_mask


import pandas as pd
