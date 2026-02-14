"""
DataFactory for SSMD-UNet Phase 1: EyePACS Dataset
Handles image loading, crop, blur, resize, and normalization
"""

import os
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EyePACSDataFactory:
    """
    Manages EyePACS dataset metadata and file paths
    Respects AGENT.md: No image loading, no tensors, only metadata
    """
    
    def __init__(self, data_root: str):
        """
        Initialize factory with dataset root
        
        Args:
            data_root: Path to eye_pacs/data/train containing images
        """
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise ValueError(f"Data root not found: {data_root}")
        
        self.df = None
    
    def scan_dataset(self) -> pd.DataFrame:
        """
        Scan directory structure and create metadata DataFrame
        
        Returns:
            DataFrame with columns: [image_path, image_name]
        """
        image_paths = []
        image_names = []
        
        # Search for all jpg files
        jpg_files = sorted(self.data_root.glob("*.jpg"))
        jpeg_files = sorted(self.data_root.glob("*.jpeg"))
        all_files = jpg_files + jpeg_files
        
        logger.info(f"Found {len(all_files)} images in {self.data_root}")
        
        for img_path in all_files:
            image_paths.append(str(img_path))
            image_names.append(img_path.stem)
        
        self.df = pd.DataFrame({
            'image_path': image_paths,
            'image_name': image_names
        })
        
        logger.info(f"Created metadata DataFrame with {len(self.df)} rows")
        return self.df
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get the metadata DataFrame"""
        if self.df is None:
            return self.scan_dataset()
        return self.df
    
    def save_metadata(self, csv_path: str):
        """Save metadata to CSV"""
        if self.df is None:
            self.scan_dataset()
        self.df.to_csv(csv_path, index=False)
        logger.info(f"Metadata saved to {csv_path}")
    
    def load_metadata(self, csv_path: str):
        """Load metadata from CSV"""
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded metadata from {csv_path}: {len(self.df)} images")
        return self.df


class EyePACSDataset:
    """
    PyTorch Dataset for EyePACS
    Handles image loading, preprocessing (crop, blur, resize, normalize)
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_size: int = 512,
        apply_blur: bool = True,
        blur_kernel: int = 5,
        normalize: bool = True,
    ):
        """
        Initialize dataset
        
        Args:
            dataframe: DataFrame from EyePACSDataFactory with image_path column
            image_size: Target image size (default: 512)
            apply_blur: Apply Gaussian blur (default: True)
            blur_kernel: Gaussian blur kernel size (default: 5)
            normalize: Normalize to [0, 1] (default: True)
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_size = image_size
        self.apply_blur = apply_blur
        self.blur_kernel = blur_kernel
        self.normalize = normalize
    
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get preprocessed image
        
        Returns:
            dict with keys: 'image', 'image_path'
        """
        row = self.dataframe.iloc[idx]
        image_path = row['image_path']
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocessing pipeline
        image = self._crop_image(image)
        
        if self.apply_blur:
            image = self._apply_blur(image)
        
        image = self._resize_image(image)
        
        if self.normalize:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
        
        # Convert to CHW format (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        return {
            'image': image,
            'image_path': image_path,
            'image_name': row.get('image_name', os.path.basename(image_path))
        }
    
    def _crop_image(self, image: np.ndarray) -> np.ndarray:
        """
        Remove black borders from retinal image
        
        Args:
            image: BGR image
        
        Returns:
            Cropped image
        """
        # Convert to grayscale for border detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Find non-black pixels
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding rect of largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add small margin
            margin = 5
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image.shape[1] - x, w + 2 * margin)
            h = min(image.shape[0] - y, h + 2 * margin)
            
            return image[y:y+h, x:x+w]
        
        return image
    
    def _apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(image, (self.blur_kernel, self.blur_kernel), 0)
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image maintaining aspect ratio with padding"""
        h, w = image.shape[:2]
        max_dim = max(h, w)
        
        # Calculate scale
        scale = self.image_size / max_dim
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Pad to exact size
        if new_h != self.image_size or new_w != self.image_size:
            pad_h = self.image_size - new_h
            pad_w = self.image_size - new_w
            
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            image = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        
        return image


if __name__ == "__main__":
    # Test DataFactory
    print("Testing EyePACS DataFactory...")
    factory = EyePACSDataFactory("/home/lucas/mestrado/tapi_inrid/eye_pacs/data/train")
    df = factory.scan_dataset()
    print(f"Created DataFrame: {df.shape}")
    print(df.head())
    
    # Test Dataset
    print("\nTesting EyePACS Dataset...")
    dataset = EyePACSDataset(df, image_size=512, apply_blur=True, normalize=True)
    print(f"Dataset length: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Image dtype: {sample['image'].dtype}")
    print(f"Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
