#!/usr/bin/env python3
"""
Experiment 12: Green Channel Enhanced Input

Simple modification: Boost green channel weight (1.3x) before training.

Rationale:
- Hemorrhages show better contrast in green channel
- Keeps 3 channels (preserves ImageNet pretrained weights)
- Minimal change from baseline

Expected: +1-2% on hemorrhage Dice
"""

import sys
import json
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset_green import ROPDatasetGreenEnhanced
from train_and_val_worker import TrainAndEvalWorker
from utils.utils import set_seed


def main():
    print("="*80)
    print("EXPERIMENT 12: Green Channel Enhanced Input")
    print("="*80)
    print()
    print("Baseline (B4 + RGB): 0.6428 Test Dice")
    print("Modification: Green channel weighted 1.3x (R=0.9x, G=1.3x, B=0.9x)")
    print("Target: Improve hemorrhage Dice by 1-2%")
    print()
    print("Rationale:")
    print("  - Hemorrhages have best contrast in green channel")
    print("  - Keeps 3 channels (preserves ImageNet weights)")
    print("  - Zero architecture change")
    print()
    print("="*80)
    print()
    
    # Config
    config = Config()
    config.model_name = "unet"
    config.encoder_name = "efficientnet-b4"
    config.encoder_weights = "imagenet"
    config.resolution = 512
    config.batch_size = 8
    config.num_epochs = 50
    config.learning_rate = 1e-4
    
    config.checkpoint_dir = os.path.join(config.output_dir, "checkpoints", "green_channel")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    set_seed(config.random_state)
    
    # Load data
    data_factory = DataFactory(config)
    train_df, test_df = data_factory.create_metadata_dataframe()
    
    print(f"✓ Training images: {len(train_df)}")
    print(f"✓ Test images: {len(test_df)}")
    print()
    
    # Load frozen splits
    splits_path = Path("outputs/cv_splits.json")
    with open(splits_path) as f:
        cv_data = json.load(f)
    
    print(f"✓ Loaded frozen splits")
    print()
    
    # Initialize trainer
    trainer = TrainAndEvalWorker(config)
    
    # Storage
    fold_results = []
    saved_model_paths = []
    
    # 5-fold CV
    for fold_idx in range(5):
        print("\n" + "="*80)
        print(f"FOLD {fold_idx + 1}/5")
        print("="*80)
        
        fold_data = cv_data['folds'][fold_idx]
        train_indices = fold_data['train_indices']
        val_indices = fold_data['val_indices']
        
        print(f"Train samples: {len(train_indices)}")
        print(f"Val samples: {len(val_indices)}")
        
        train_fold_df = train_df.iloc[train_indices].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_indices].reset_index(drop=True)
        
        # Use green-enhanced dataset
        train_dataset = ROPDatasetGreenEnhanced(
            dataframe=train_fold_df,
            config=config,
            is_train=True,
            green_weight=1.3  # Boost green 30%
        )
        
        val_dataset = ROPDatasetGreenEnhanced(
            dataframe=val_fold_df,
            config=config,
            is_train=False,
            green_weight=1.3
        )
        
        # Train
        model, history = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            fold=fold_idx + 1
        )
        
        best_dice = max(history['val_dice'])
        best_epoch = history['val_dice'].index(best_dice) + 1
        
        model_path = os.path.join(config.checkpoint_dir, f"best_model_fold{fold_idx + 1}.pth")
        saved_model_paths.append(model_path)
        
        fold_results.append({
            'fold': fold_idx + 1,
            'best_val_dice': best_dice,
            'best_epoch': best_epoch
        })
        
        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  Best Val Dice: {best_dice:.4f}")
        print(f"  Best Epoch: {best_epoch}")
    
    # CV results
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS")
    print("="*80)
    
    mean_dice = sum(f['best_val_dice'] for f in fold_results) / len(fold_results)
    std_dice = (sum((f['best_val_dice'] - mean_dice)**2 for f in fold_results) / len(fold_results))**0.5
    
    print(f"\nCV Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Baseline CV:  0.5444")
    print(f"Difference:   {mean_dice - 0.5444:+.4f}")
    
    for result in fold_results:
        print(f"  Fold {result['fold']}: {result['best_val_dice']:.4f} (epoch {result['best_epoch']})")
    
    # Test evaluation
    print("\n" + "="*80)
    print("FINAL TEST SET EVALUATION (Ensemble + TTA)")
    print("="*80)
    
    test_dataset = ROPDatasetGreenEnhanced(
        dataframe=test_df,
        config=config,
        is_train=False,
        green_weight=1.3
    )
    
    test_results = trainer.evaluate_ensemble(
        test_dataset=test_dataset,
        model_paths=saved_model_paths,
        use_tta=True
    )
    
    print(f"\nMean Dice: {test_results['mean_dice']:.4f}")
    print(f"Exudates:  {test_results['dice_exudates']:.4f}")
    print(f"Hemorrhages: {test_results['dice_haemorrhages']:.4f}")
    
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Baseline (RGB normal):   0.6428")
    print(f"This (Green 1.3x):       {test_results['mean_dice']:.4f}")
    print(f"Difference:              {test_results['mean_dice'] - 0.6428:+.4f}")
    
    if test_results['mean_dice'] >= 0.650:
        print("\n🎯 TARGET ACHIEVED: Dice ≥ 0.65 (paper-level)")
    elif test_results['mean_dice'] > 0.6428:
        print(f"\n✅ IMPROVEMENT over baseline")
    else:
        print(f"\n❌ Green enhancement didn't help")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
