"""
Fast EyePACS dataset that loads pre-processed tensors
Much faster than on-the-fly preprocessing
"""

import torch
from pathlib import Path
from torch.utils.data import Dataset
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class EyePACSFastDataset(Dataset):
    """Load pre-processed tensor files (.pt)"""
    
    def __init__(self, data_dir: str = "/home/lucas/mestrado/tapi_inrid/eye_pacs/data/train_preprocessed"):
        """
        Args:
            data_dir: Directory containing preprocessed .pt files
        """
        self.data_dir = Path(data_dir)
        
        # Find all .pt files
        self.pt_files = sorted(self.data_dir.glob("*.pt"))
        
        if len(self.pt_files) == 0:
            raise ValueError(f"No .pt files found in {self.data_dir}")
        
        logger.info(f"Found {len(self.pt_files)} preprocessed images in {self.data_dir}")
    
    def __len__(self):
        return len(self.pt_files)
    
    def __getitem__(self, idx):
        """Load tensor from disk"""
        pt_path = self.pt_files[idx]
        # Use map_location='cpu' to avoid GPU overhead during loading
        img_tensor = torch.load(pt_path, weights_only=True, map_location='cpu')
        
        return {
            'image': img_tensor,
            'name': pt_path.stem
        }


if __name__ == "__main__":
    dataset = EyePACSFastDataset()
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shape: {sample['image'].shape}")
    print(f"Sample dtype: {sample['image'].dtype}")
    print(f"Sample range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
