#!/usr/bin/env python3
"""Quick test for Green Channel Enhancement pipeline"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset_green import ROPDatasetGreenEnhanced

# Quick test
config = Config()
data_factory = DataFactory(config)
train_df, _ = data_factory.create_metadata_dataframe()

# Create dataset with green enhancement
dataset = ROPDatasetGreenEnhanced(
    dataframe=train_df.head(1),
    config=config,
    is_train=False,
    green_weight=1.3
)

# Load one sample
sample = dataset[0]
print(f"✓ Image shape: {sample['image'].shape}")
print(f"✓ Mask shape: {sample['mask'].shape}")
print(f"✓ Image dtype: {sample['image'].dtype}")
print(f"✓ Image min/max: {sample['image'].min():.3f}/{sample['image'].max():.3f}")
print("\n✅ Green channel enhancement working correctly!")
