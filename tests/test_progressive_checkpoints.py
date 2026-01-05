#!/usr/bin/env python3
"""Test progressive training checkpoints quickly before full training."""

import sys
import os
from pathlib import Path

sys.path.append(str(Path.cwd()))

from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset import ROPDataset
from train_and_val_worker import TrainAndEvalWorker
from utils.utils import set_seed


def main():
    print("="*80)
    print("TESTING PROGRESSIVE TRAINING CHECKPOINTS")
    print("="*80)
    print()
    
    # Initialize config
    config = Config()
    config.model_name = "unet"
    config.encoder_name = "efficientnet-b4"
    config.encoder_weights = "imagenet"
    config.resolution = 512
    config.batch_size = 8
    
    set_seed(config.random_state)
    
    # Create data
    print("Loading dataset...")
    data_factory = DataFactory(config)
    train_df, test_df = data_factory.create_metadata_dataframe()
    print(f"✓ Test images: {len(test_df)}")
    print()
    
    # Check if checkpoints exist
    checkpoint_dir = "outputs/checkpoints/progressive_training"
    model_paths = []
    
    for fold_idx in range(5):
        path = f"{checkpoint_dir}/fold_{fold_idx}_best.pth"
        if os.path.exists(path):
            print(f"✓ Found: {path}")
            model_paths.append(path)
        else:
            print(f"✗ Missing: {path}")
    
    if len(model_paths) != 5:
        print(f"\n❌ ERROR: Only found {len(model_paths)}/5 checkpoints")
        print("Need to train first!")
        return 1
    
    print(f"\n✓ All 5 checkpoints found!")
    print()
    
    # Create test dataset
    test_dataset = ROPDataset(
        dataframe=test_df,
        config=config,
        is_train=False
    )
    
    # Initialize trainer
    trainer = TrainAndEvalWorker(config)
    
    # Evaluate with ensemble + TTA
    print("="*80)
    print("EVALUATING TEST SET (Ensemble + TTA)")
    print("="*80)
    print()
    
    test_results = trainer.evaluate_ensemble(
        test_dataset=test_dataset,
        model_paths=model_paths,
        use_tta=True
    )
    
    # Display results
    print("\n=== Test Results (Ensemble + TTA) ===")
    print(f"Mean Dice: {test_results['dice']:.4f}")
    print(f"Mean IoU: {test_results['iou']:.4f}")
    print(f"Exudates - Dice: {test_results['per_class_dice'][0]:.4f}, IoU: {test_results['per_class_iou'][0]:.4f}")
    print(f"Haemorrhages - Dice: {test_results['per_class_dice'][1]:.4f}, IoU: {test_results['per_class_iou'][1]:.4f}")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINE")
    print("="*80)
    baseline = 0.6448
    obtained = test_results['dice']
    diff = obtained - baseline
    diff_pct = (diff / baseline) * 100
    
    print(f"Baseline (Ensemble+TTA):  {baseline:.4f}")
    print(f"Progressive Training:     {obtained:.4f}")
    print(f"Difference: {diff:+.4f} ({diff_pct:+.2f}%)")
    
    if obtained > baseline:
        print(f"\n✅ SUCCESS! Improved by {diff_pct:.2f}%")
    else:
        print(f"\n❌ Did not beat baseline (worse by {abs(diff_pct):.2f}%)")
    
    print("\n" + "="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
