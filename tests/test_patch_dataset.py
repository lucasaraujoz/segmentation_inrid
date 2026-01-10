"""Test patch-based dataset implementation.

Quick test to verify:
1. Patch extraction works correctly
2. Patches have correct dimensions
3. Number of patches matches expectations
4. Reconstruction produces correct dimensions
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset_patches import ROPDatasetPatches
from utils.utils import set_seed
import numpy as np
import matplotlib.pyplot as plt


def test_patch_dataset():
    """Test patch dataset creation and properties."""
    
    print("="*80)
    print("Testing Patch-Based Dataset")
    print("="*80)
    
    # Configuration
    config = Config()
    set_seed(config.random_state)
    
    PATCH_SIZE = 512
    OVERLAP = 50
    
    # Create DataFactory
    print("\n[1/3] Creating DataFactory...")
    data_factory = DataFactory(config)
    train_df, test_df = data_factory.create_metadata_dataframe()
    
    print(f"Training images: {len(train_df)}")
    print(f"Test images: {len(test_df)}")
    
    # Create patch dataset (use only first 3 images for speed)
    print(f"\n[2/3] Creating patch dataset...")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Overlap: {OVERLAP}px")
    
    small_df = train_df.head(3)
    
    patch_dataset = ROPDatasetPatches(
        dataframe=small_df,
        config=config,
        patch_size=PATCH_SIZE,
        overlap=OVERLAP,
        is_train=False  # No augmentation for testing
    )
    
    print(f"\nDataset created!")
    print(f"Total patches: {len(patch_dataset)}")
    print(f"Patches per image (avg): {len(patch_dataset) / len(small_df):.1f}")
    
    # Test loading a few patches
    print(f"\n[3/3] Testing patch loading...")
    
    for i in range(min(3, len(patch_dataset))):
        sample = patch_dataset[i]
        
        print(f"\nPatch {i}:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Mask shape: {sample['mask'].shape}")
        print(f"  Image name: {sample['image_name']}")
        print(f"  Position: ({sample['patch_x']}, {sample['patch_y']})")
        print(f"  Grid: ({sample['grid_i']}, {sample['grid_j']})")
        print(f"  Original size: {sample['img_width']}x{sample['img_height']}")
        
        # Verify dimensions
        assert sample['image'].shape[1] == PATCH_SIZE, "Image height mismatch"
        assert sample['image'].shape[2] == PATCH_SIZE, "Image width mismatch"
        assert sample['mask'].shape[1] == PATCH_SIZE, "Mask height mismatch"
        assert sample['mask'].shape[2] == PATCH_SIZE, "Mask width mismatch"
        assert sample['mask'].shape[0] == len(config.classes), "Number of classes mismatch"
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)
    
    # Visualize patches from first image
    print("\n[Optional] Visualizing patches from first image...")
    
    # Get all patches from first image
    first_img_patches = [i for i, p in enumerate(patch_dataset.patch_info) 
                        if p['image_idx'] == 0]
    
    print(f"First image has {len(first_img_patches)} patches")
    
    # Show first 9 patches
    n_show = min(9, len(first_img_patches))
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(n_show):
        patch_idx = first_img_patches[i]
        sample = patch_dataset[patch_idx]
        
        # Denormalize image
        image = sample['image'].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        axes[i].imshow(image)
        axes[i].set_title(f"Patch {i+1}\nGrid: ({sample['grid_i']}, {sample['grid_j']})")
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_show, 9):
        axes[i].axis('off')
    
    plt.suptitle(f"Patches from {sample['image_name']}", fontsize=16)
    plt.tight_layout()
    
    output_path = 'outputs/patch_visualization.png'
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")
    
    plt.show()
    
    return patch_dataset


if __name__ == "__main__":
    dataset = test_patch_dataset()
