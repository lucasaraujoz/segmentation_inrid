"""DataFactory for managing dataset metadata and splits.

This module is responsible for:
- Traversing the dataset directory tree
- Mapping image and mask paths
- Creating a structured pandas DataFrame
- Preparing data splits for cross-validation

NEVER loads actual images or tensors, only manages metadata.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.model_selection import GroupKFold
import re


class DataFactory:
    """Factory class for managing dataset metadata and creating splits."""
    
    def __init__(self, config):
        """Initialize DataFactory with configuration.
        
        Args:
            config: Configuration object containing dataset paths and settings
        """
        self.config = config
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        
    def create_metadata_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create structured DataFrames with image and mask paths.
        
        Returns:
            Tuple of (train_df, test_df) containing metadata
        """
        print("Creating metadata DataFrames...")
        
        # Process training set
        self.train_df = self._process_split(
            images_dir=self.config.original_images_train,
            gt_dir=self.config.segmentation_gt_train,
            split_name="train"
        )
        
        # Process testing set
        self.test_df = self._process_split(
            images_dir=self.config.original_images_test,
            gt_dir=self.config.segmentation_gt_test,
            split_name="test"
        )
        
        print(f"Train set: {len(self.train_df)} images")
        print(f"Test set: {len(self.test_df)} images")
        
        return self.train_df, self.test_df
    
    def _process_split(self, images_dir: str, gt_dir: str, split_name: str) -> pd.DataFrame:
        """Process a single split (train or test).
        
        Args:
            images_dir: Directory containing original images
            gt_dir: Directory containing segmentation ground truths
            split_name: Name of the split ('train' or 'test')
            
        Returns:
            DataFrame with metadata for this split
        """
        # Get all image files
        image_files = sorted(list(Path(images_dir).glob("*.jpg")))
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {images_dir}")
        
        records = []
        
        for img_path in image_files:
            img_name = img_path.stem
            
            # Extract patient ID from filename (e.g., "IDRiD_01" -> "01")
            patient_id = self._extract_patient_id(img_name)
            
            record = {
                "image_path": str(img_path),
                "image_name": img_name,
                "split": split_name,
                "patient_id": patient_id
            }
            
            # For each class, find corresponding mask files
            for class_name in self.config.classes:
                mask_paths = self._find_masks_for_class(
                    img_name, class_name, gt_dir
                )
                record[f"mask_{class_name}_paths"] = mask_paths
                record[f"has_{class_name}"] = len(mask_paths) > 0
            
            records.append(record)
        
        df = pd.DataFrame(records)
        return df
    
    def _find_masks_for_class(
        self, image_name: str, class_name: str, gt_dir: str
    ) -> List[str]:
        """Find all mask files for a given image and class.
        
        Args:
            image_name: Name of the image (without extension)
            class_name: Class name (e.g., 'exudates', 'haemorrhages')
            gt_dir: Ground truth directory
            
        Returns:
            List of paths to mask files (can be multiple for exudates)
        """
        mask_paths = []
        
        # Get folders that correspond to this class
        folders = self.config.get_class_folders(class_name)
        
        for folder in folders:
            folder_path = Path(gt_dir) / folder
            
            if not folder_path.exists():
                continue
            
            # Look for masks matching the image name
            mask_files = list(folder_path.glob(f"{image_name}*.tif")) + \
                        list(folder_path.glob(f"{image_name}*.tiff"))
            
            mask_paths.extend([str(p) for p in mask_files])
        
        return mask_paths
    
    def _extract_patient_id(self, image_name: str) -> str:
        """Extract patient ID from image filename.
        
        Args:
            image_name: Image filename (e.g., "IDRiD_01")
            
        Returns:
            Patient ID string
        """
        # Try to extract numeric part from filename
        match = re.search(r'(\d+)', image_name)
        if match:
            return match.group(1)
        return image_name
    
    def prepare_data_for_cross_validation(
        self
    ) -> Tuple[List[Tuple[List[int], List[int]]], List[str], pd.DataFrame]:
        """Prepare data splits for cross-validation.
        
        Operates only on the training set.
        Respects patient grouping to avoid data leakage.
        
        Returns:
            Tuple of:
                - List of (train_indices, val_indices) for each fold
                - List of patient IDs
                - Test DataFrame (fixed, never used in CV)
        """
        if self.train_df is None or self.test_df is None:
            raise ValueError("Must call create_metadata_dataframe() first")
        
        # Get patient IDs for grouping
        patient_ids = self.train_df["patient_id"].values
        
        # Create GroupKFold splitter
        gkf = GroupKFold(n_splits=self.config.n_folds)
        
        # Generate splits
        splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(self.train_df, groups=patient_ids)
        ):
            splits.append((train_idx.tolist(), val_idx.tolist()))
            print(f"Fold {fold_idx + 1}: {len(train_idx)} train, {len(val_idx)} val")
        
        return splits, patient_ids.tolist(), self.test_df
    
    def get_train_dataframe(self) -> pd.DataFrame:
        """Get the training DataFrame.
        
        Returns:
            Training DataFrame
        """
        if self.train_df is None:
            raise ValueError("Must call create_metadata_dataframe() first")
        return self.train_df
    
    def get_test_dataframe(self) -> pd.DataFrame:
        """Get the test DataFrame.
        
        Returns:
            Test DataFrame
        """
        if self.test_df is None:
            raise ValueError("Must call create_metadata_dataframe() first")
        return self.test_df
    
    def get_class_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get distribution of classes in the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            DataFrame with class distribution statistics
        """
        distribution = {}
        
        for class_name in self.config.classes:
            has_col = f"has_{class_name}"
            if has_col in df.columns:
                count = df[has_col].sum()
                percentage = (count / len(df)) * 100
                distribution[class_name] = {
                    "count": count,
                    "percentage": percentage
                }
        
        return pd.DataFrame(distribution).T
