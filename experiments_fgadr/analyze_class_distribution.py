#!/usr/bin/env python3
"""
Analyze FGADR class distribution (pixel counts per class)

This script calculates total pixels for each class across the entire dataset
to identify potential class imbalance issues.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data_factory.fgadr_multiclass_dataset import create_fgadr_dataframe


def analyze_pixel_distribution():
    """Analyze pixel distribution across all images"""
    
    data_dir = Path('/home/lucas/fgadr/Seg-set')
    
    # Get all images
    df = create_fgadr_dataframe()
    print(f"\nTotal images: {len(df)}")
    
    # Initialize counters
    total_pixels = 0
    exudates_pixels = 0
    hemorrhage_pixels = 0
    both_classes_pixels = 0  # Co-occurrence
    
    images_with_exudates = 0
    images_with_hemorrhage = 0
    images_with_both = 0
    images_with_none = 0
    
    print("\nProcessing masks...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['filename']
        
        # Load Hard Exudate mask
        hard_ex_path = data_dir / 'HardExudate_Masks' / filename
        soft_ex_path = data_dir / 'SoftExudate_Masks' / filename
        hemorrhage_path = data_dir / 'Hemohedge_Masks' / filename
        
        # Load masks
        hard_ex = cv2.imread(str(hard_ex_path), cv2.IMREAD_GRAYSCALE)
        soft_ex = cv2.imread(str(soft_ex_path), cv2.IMREAD_GRAYSCALE)
        hemorrhage = cv2.imread(str(hemorrhage_path), cv2.IMREAD_GRAYSCALE)
        
        if hard_ex is None or soft_ex is None or hemorrhage is None:
            continue
        
        # Combine exudates (Hard OR Soft)
        exudates = np.maximum(hard_ex, soft_ex)
        
        # Binarize
        exudates = (exudates > 0).astype(np.uint8)
        hemorrhage = (hemorrhage > 0).astype(np.uint8)
        
        # Count pixels
        img_total = exudates.size
        img_exudates = exudates.sum()
        img_hemorrhage = hemorrhage.sum()
        img_both = (exudates & hemorrhage).sum()  # Co-occurrence
        
        total_pixels += img_total
        exudates_pixels += img_exudates
        hemorrhage_pixels += img_hemorrhage
        both_classes_pixels += img_both
        
        # Count images with each class
        if img_exudates > 0:
            images_with_exudates += 1
        if img_hemorrhage > 0:
            images_with_hemorrhage += 1
        if img_exudates > 0 and img_hemorrhage > 0:
            images_with_both += 1
        if img_exudates == 0 and img_hemorrhage == 0:
            images_with_none += 1
    
    # Calculate statistics
    print("\n" + "="*70)
    print("PIXEL DISTRIBUTION ANALYSIS")
    print("="*70)
    
    print(f"\nTotal pixels analyzed: {total_pixels:,}")
    print(f"Total images: {len(df)}")
    
    print("\n--- Class Distribution (Pixels) ---")
    exudates_pct = (exudates_pixels / total_pixels) * 100
    hemorrhage_pct = (hemorrhage_pixels / total_pixels) * 100
    background_pct = 100 - exudates_pct - hemorrhage_pct + (both_classes_pixels / total_pixels) * 100
    
    print(f"Exudates:   {exudates_pixels:12,} pixels ({exudates_pct:.4f}%)")
    print(f"Hemorrhage: {hemorrhage_pixels:12,} pixels ({hemorrhage_pct:.4f}%)")
    print(f"Co-occurrence: {both_classes_pixels:12,} pixels ({(both_classes_pixels/total_pixels)*100:.4f}%)")
    print(f"Background: ~{background_pct:.4f}%")
    
    # Class imbalance ratio
    if hemorrhage_pixels > 0:
        ratio = exudates_pixels / hemorrhage_pixels
        print(f"\nExudates/Hemorrhage Ratio: {ratio:.2f}:1")
    
    print("\n--- Image-level Distribution ---")
    print(f"Images with Exudates:   {images_with_exudates:4d} ({images_with_exudates/len(df)*100:.1f}%)")
    print(f"Images with Hemorrhage: {images_with_hemorrhage:4d} ({images_with_hemorrhage/len(df)*100:.1f}%)")
    print(f"Images with BOTH:       {images_with_both:4d} ({images_with_both/len(df)*100:.1f}%)")
    print(f"Images with NONE:       {images_with_none:4d} ({images_with_none/len(df)*100:.1f}%)")
    
    print("\n--- Recommendations ---")
    if ratio > 2 or ratio < 0.5:
        print("⚠️  SIGNIFICANT CLASS IMBALANCE DETECTED!")
        print(f"   Consider using class weights in loss function.")
        print(f"   Suggested weights (inverse frequency):")
        total_lesion_pixels = exudates_pixels + hemorrhage_pixels - both_classes_pixels
        w_exudates = total_lesion_pixels / (2 * exudates_pixels)
        w_hemorrhage = total_lesion_pixels / (2 * hemorrhage_pixels)
        print(f"   - Exudates weight:   {w_exudates:.4f}")
        print(f"   - Hemorrhage weight: {w_hemorrhage:.4f}")
    else:
        print("✅ Classes are relatively balanced!")
    
    # Save statistics
    stats = {
        'total_images': len(df),
        'total_pixels': int(total_pixels),
        'exudates_pixels': int(exudates_pixels),
        'hemorrhage_pixels': int(hemorrhage_pixels),
        'co_occurrence_pixels': int(both_classes_pixels),
        'exudates_percentage': float(exudates_pct),
        'hemorrhage_percentage': float(hemorrhage_pct),
        'class_ratio': float(ratio) if hemorrhage_pixels > 0 else None,
        'images_with_exudates': int(images_with_exudates),
        'images_with_hemorrhage': int(images_with_hemorrhage),
        'images_with_both': int(images_with_both),
        'images_with_none': int(images_with_none)
    }
    
    output_path = Path(__file__).parent.parent / 'outputs' / 'fgadr_class_distribution.json'
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Statistics saved to: {output_path}")
    print("="*70)
    
    return stats


if __name__ == "__main__":
    analyze_pixel_distribution()
