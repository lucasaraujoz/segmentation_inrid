#!/usr/bin/env python3
"""
Quick evaluation of B1 test set (training already done)
"""

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset import ROPDataset
from train_and_val_worker import TrainAndEvalWorker
from utils.utils import set_seed


def main():
    print("="*80)
    print("EVALUATING B1 TEST SET (Ensemble + TTA)")
    print("="*80)
    
    config = Config()
    config.model_name = "unet"
    config.encoder_name = "efficientnet-b1"
    config.encoder_weights = "imagenet"
    config.resolution = 512
    config.batch_size = 8
    config.checkpoint_dir = os.path.join(config.output_dir, "checkpoints", "efficientnet_b1")
    
    set_seed(config.random_state)
    
    # Load data
    data_factory = DataFactory(config)
    train_df, test_df = data_factory.create_metadata_dataframe()
    
    print(f"Test images: {len(test_df)}")
    
    # Model paths
    saved_model_paths = [
        os.path.join(config.checkpoint_dir, f"best_model_fold{i}.pth")
        for i in range(1, 6)
    ]
    
    # Check all exist
    for path in saved_model_paths:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found")
            return 1
        print(f"✓ Found: {path}")
    
    print()
    
    # Create test dataset
    test_dataset = ROPDataset(
        dataframe=test_df,
        config=config,
        is_train=False
    )
    
    # Initialize trainer
    trainer = TrainAndEvalWorker(config)
    
    # Evaluate
    print("Running ensemble evaluation with TTA...")
    test_results = trainer.evaluate_ensemble(
        test_dataset=test_dataset,
        model_paths=saved_model_paths,
        use_tta=True
    )
    
    # Results
    print("\n" + "="*80)
    print("RESULTS: EfficientNet-B1")
    print("="*80)
    print(f"Mean Dice: {test_results['mean_dice']:.4f}")
    print(f"Mean IoU: {test_results['mean_iou']:.4f}")
    print(f"Exudates - Dice: {test_results['dice_exudates']:.4f}, IoU: {test_results['iou_exudates']:.4f}")
    print(f"Haemorrhages - Dice: {test_results['dice_haemorrhages']:.4f}, IoU: {test_results['iou_haemorrhages']:.4f}")
    
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Baseline (B4):  0.6448")
    print(f"This (B1):      {test_results['mean_dice']:.4f}")
    diff = test_results['mean_dice'] - 0.6448
    print(f"Difference:     {diff:+.4f}")
    
    if test_results['mean_dice'] > 0.6448:
        print("\n✅ B1 > B4")
    elif test_results['mean_dice'] > 0.6400:
        print("\n≈ B1 ≈ B4")
    else:
        print("\n❌ B1 < B4")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
