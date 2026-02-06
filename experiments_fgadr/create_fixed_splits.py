#!/usr/bin/env python3
"""
Generate fixed train/val/test splits for FGADR dataset

Creates a JSON file with reproducible splits to ensure consistency across experiments.
Similar to cv_splits.json and frozen_cv_splits.json used for IDRiD.
"""

import sys
from pathlib import Path
import numpy as np
import json
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data_factory.fgadr_multiclass_dataset import create_fgadr_dataframe


def create_fgadr_splits(
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
):
    """
    Create fixed train/val/test splits for FGADR
    
    Args:
        train_ratio: Proportion for training set (default: 0.70)
        val_ratio: Proportion for validation set (default: 0.15)
        test_ratio: Proportion for test set (default: 0.15)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with train/val/test filenames
    """
    
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Load dataset
    print("Loading FGADR dataset...")
    df = create_fgadr_dataframe()
    
    print(f"Total images: {len(df)}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    
    # Get all filenames
    all_files = df['filename'].tolist()
    
    # Set random seed
    np.random.seed(random_seed)
    
    # First split: train vs (val+test)
    train_files, temp_files = train_test_split(
        all_files,
        test_size=(val_ratio + test_ratio),
        random_state=random_seed,
        shuffle=True
    )
    
    # Second split: val vs test
    val_files, test_files = train_test_split(
        temp_files,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_seed,
        shuffle=True
    )
    
    # Create splits dictionary
    splits = {
        'dataset': 'FGADR',
        'total_images': len(df),
        'random_seed': random_seed,
        'split_ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        },
        'split_counts': {
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files)
        },
        'train': sorted(train_files),
        'val': sorted(val_files),
        'test': sorted(test_files)
    }
    
    # Print summary
    print("\n" + "="*70)
    print("FGADR DATASET SPLITS")
    print("="*70)
    print(f"\nTotal images: {len(df)}")
    print(f"\nSplit ratios:")
    print(f"  Train: {train_ratio:.1%} ({len(train_files)} images)")
    print(f"  Val:   {val_ratio:.1%} ({len(val_files)} images)")
    print(f"  Test:  {test_ratio:.1%} ({len(test_files)} images)")
    print(f"\nRandom seed: {random_seed}")
    
    # Verify no overlap
    train_set = set(train_files)
    val_set = set(val_files)
    test_set = set(test_files)
    
    assert len(train_set & val_set) == 0, "Train/Val overlap!"
    assert len(train_set & test_set) == 0, "Train/Test overlap!"
    assert len(val_set & test_set) == 0, "Val/Test overlap!"
    
    print("\n✅ No overlap between splits")
    print("="*70)
    
    return splits


def save_splits(splits, output_path):
    """Save splits to JSON file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n✅ Splits saved to: {output_path}")


if __name__ == "__main__":
    # Generate splits
    splits = create_fgadr_splits(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    # Save to outputs
    output_path = Path(__file__).parent.parent / 'outputs' / 'fgadr_splits.json'
    save_splits(splits, output_path)
    
    print("\n🎯 Usage in experiments:")
    print("   import json")
    print("   with open('outputs/fgadr_splits.json') as f:")
    print("       splits = json.load(f)")
    print("   train_files = splits['train']")
    print("   val_files = splits['val']")
    print("   test_files = splits['test']")
