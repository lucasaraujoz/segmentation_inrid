#!/usr/bin/env python3
"""
Evaluate test set with Ensemble + TTA using saved models.
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
    print("TEST SET EVALUATION: Ensemble + TTA")
    print("="*80)
    print()
    
    # Initialize config
    config = Config()
    config.model_name = "unet"
    config.encoder_name = "efficientnet-b4"
    config.encoder_weights = "imagenet"
    config.resolution = 512
    config.batch_size = 8
    
    # Set checkpoint directory to where models are saved
    checkpoint_dir = "outputs/checkpoints/baseline_verify"
    
    # Set seed
    set_seed(config.random_state)
    
    # Create data factory
    print("Loading dataset...")
    data_factory = DataFactory(config)
    
    # Create metadata
    train_df, test_df = data_factory.create_metadata_dataframe()
    print(f"✓ Test images: {len(test_df)}")
    print()
    
    # Get saved model paths
    model_paths = []
    for fold_idx in range(1, 6):
        model_path = os.path.join(checkpoint_dir, f"best_model_fold{fold_idx}.pth")
        if os.path.exists(model_path):
            model_paths.append(model_path)
            print(f"✓ Found model for fold {fold_idx}: {model_path}")
        else:
            print(f"✗ Missing model for fold {fold_idx}: {model_path}")
    
    if len(model_paths) != 5:
        print(f"\n❌ ERROR: Expected 5 models, found {len(model_paths)}")
        return 1
    
    print(f"\n✓ All 5 models found!")
    print()
    
    # Initialize trainer
    trainer = TrainAndEvalWorker(config)
    
    # Create test dataset
    test_dataset = ROPDataset(
        dataframe=test_df,
        config=config,
        is_train=False
    )
    
    print("="*80)
    print("FINAL TEST SET EVALUATION (Ensemble + TTA)")
    print("="*80)
    print()
    print(f"Using ensemble of {len(model_paths)} folds + TTA...")
    print(f"TTA transforms: {len(config.tta_transforms)}")
    print()
    
    # Evaluate with ensemble + TTA
    test_results = trainer.evaluate_ensemble(
        test_dataset=test_dataset,
        model_paths=model_paths,
        use_tta=True
    )
    
    # Display test results
    print("\n=== Test Results (Ensemble + TTA) ===")
    print(f"Mean Dice: {test_results['mean_dice']:.4f}")
    print(f"Mean IoU: {test_results['mean_iou']:.4f}")
    
    if 'dice_exudates' in test_results:
        print(f"Exudates - Dice: {test_results['dice_exudates']:.4f}, IoU: {test_results['iou_exudates']:.4f}")
        print(f"Haemorrhages - Dice: {test_results['dice_haemorrhages']:.4f}, IoU: {test_results['iou_haemorrhages']:.4f}")
    
    # Final comparison
    print("\n" + "="*80)
    print("COMPARISON WITH EXPECTED BASELINE")
    print("="*80)
    expected_test = 0.6442
    obtained_test = test_results['mean_dice']
    
    print(f"Expected (from training_efficientnet_b4_512_FINAL.log):  {expected_test:.4f}")
    print(f"Obtained (Ensemble+TTA):                                 {obtained_test:.4f}")
    
    test_diff = ((obtained_test - expected_test) / expected_test) * 100
    print(f"Difference: {test_diff:+.2f}%")
    
    if abs(test_diff) < 1.0:
        print("\n✅ BASELINE CONFIRMED: Test results match within 1%")
    elif abs(test_diff) < 3.0:
        print("\n✅ BASELINE CONSISTENT: Test results match within 3%")
    else:
        print(f"\n⚠️  Test results differ by {abs(test_diff):.2f}%")
    
    print("\n" + "="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
