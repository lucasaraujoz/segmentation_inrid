"""
FGADR 2-Class Dataset with CLAHE + Bilateral preprocessing

2 Classes (Multi-label):
- Channel 0: Exudates (Hard + Soft combined)
- Channel 1: Hemorrhage

Activation: Sigmoid (allows multi-label - pixel can have both classes)
"""

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path


class FGADRMultiClassDataset(Dataset):
    """FGADR Dataset for 2-class multi-label segmentation with CLAHE+Bilateral preprocessing
    
    Classes:
    - Channel 0: Exudates (Hard + Soft combined)
    - Channel 1: Hemorrhage
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        root_dir: str = "/home/lucas/fgadr/Seg-set",
        image_size: tuple = (512, 512),
        is_train: bool = True,
        transform: Optional[A.Compose] = None,
    ):
        """
        Args:
            dataframe: DataFrame with 'filename' column
            root_dir: Path to FGADR Seg-set directory
            image_size: Target image size (H, W)
            is_train: Whether this is training data
            transform: Optional custom transform
        """
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.is_train = is_train
        
        # Paths
        self.images_dir = self.root_dir / "Original_Images"
        
        # 2 classes: Exudates (Hard+Soft combined) and Hemorrhage
        # We'll load these masks and combine Hard+Soft
        self.exudate_folders = ['HardExudate_Masks', 'SoftExudate_Masks']
        self.hemorrhage_folder = 'Hemohedge_Masks'
        
        self.num_classes = 2  # Exudates, Hemorrhage
        
        # CLAHE for preprocessing
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Setup transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform()
    
    def _get_default_transform(self) -> A.Compose:
        """Get default augmentation pipeline"""
        if self.is_train:
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
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
                    A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                    A.GridDistortion(p=1.0),
                    A.OpticalDistortion(distort_limit=0.5, p=1.0),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.MultiplicativeNoise(multiplier=[0.8, 1.2], p=1.0),
                ], p=0.2),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE preprocessing (from notebook analysis)"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def apply_bilateral(self, image: np.ndarray) -> np.ndarray:
        """Apply Bilateral Filter (from notebook analysis)"""
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE + Bilateral pipeline"""
        # 1. CLAHE for contrast enhancement
        image = self.apply_clahe(image)
        # 2. Bilateral filter for noise reduction (preserves edges)
        image = self.apply_bilateral(image)
        return image
    
    def load_mask(self, filename: str, class_idx: int) -> np.ndarray:
        """Load mask for a specific class
        
        Args:
            filename: Image filename
            class_idx: 0 for Exudates, 1 for Hemorrhage
        """
        h, w = self.image_size
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if class_idx == 0:
            # Exudates: combine Hard + Soft
            for folder_name in self.exudate_folders:
                mask_path = self.root_dir / folder_name / filename
                if mask_path.exists():
                    class_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if class_mask is not None:
                        class_mask = cv2.resize(class_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        # Union: any pixel > 0 in either Hard or Soft
                        mask = np.maximum(mask, (class_mask > 0).astype(np.uint8))
        
        elif class_idx == 1:
            # Hemorrhage
            mask_path = self.root_dir / self.hemorrhage_folder / filename
            if mask_path.exists():
                class_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if class_mask is not None:
                    class_mask = cv2.resize(class_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = (class_mask > 0).astype(np.uint8)
        
        return mask
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor (3, H, W)
            mask: Tensor (2, H, W) - Channel 0: Exudates, Channel 1: Hemorrhage
        """
        # Get filename
        filename = self.df.loc[idx, 'filename']
        
        # Load image
        img_path = self.images_dir / filename
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply CLAHE + Bilateral preprocessing
        image = self.preprocess_image(image)
        
        # Resize image
        h, w = self.image_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Load 2 class masks
        masks = []
        for class_idx in range(self.num_classes):  # 0: Exudates, 1: Hemorrhage
            mask = self.load_mask(filename, class_idx)
            masks.append(mask)
        
        # Stack masks (H, W, 2)
        mask = np.stack(masks, axis=-1)
        
        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert mask to tensor [H, W, C] -> [C, H, W]
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        else:
            mask = mask.permute(2, 0, 1).float()
        
        return image, mask


def create_fgadr_dataframe():
    """Create metadata DataFrame for FGADR dataset"""
    fgadr_path = Path('/home/lucas/fgadr/Seg-set')
    images_path = fgadr_path / 'Original_Images'
    
    image_files = sorted(list(images_path.glob('*.png')))
    
    data = []
    for img_file in image_files:
        filename = img_file.name
        patient_id = filename.split('_')[0]
        data.append({
            'filename': filename,
            'patient_id': patient_id
        })
    
    df = pd.DataFrame(data)
    
    print(f"✓ FGADR Dataset")
    print(f"  Total images: {len(df)}")
    print(f"  Unique patients: {df['patient_id'].nunique()}")
    
    return df


if __name__ == "__main__":
    # Test dataset
    print("Testing FGADR Multi-class Dataset (2 classes)...\n")
    
    df = create_fgadr_dataframe()
    
    # Create dataset
    dataset = FGADRMultiClassDataset(
        dataframe=df.head(10),
        is_train=True
    )
    
    print(f"\nDataset: {len(dataset)} samples")
    
    # Test sample
    image, mask = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask dtype: {mask.dtype}")
    
    # Check which classes are present
    classes_present = mask.sum(dim=(1, 2)) > 0
    class_names = ['Exudates', 'Hemorrhage']
    print(f"\n  Classes present:")
    for i, (name, present) in enumerate(zip(class_names, classes_present)):
        if present:
            pixels = mask[i].sum().item()
            print(f"    {name}: {pixels:.0f} pixels")
    
    print("\n✓ Test passed!")

