#!/usr/bin/env python3
"""
Experiment: UNet + EfficientNet-B3

Same configuration as baseline (B4) but with B3 encoder.
Uses frozen CV splits for exact comparison.
"""

import sys
import json
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset import ROPDataset
from train_and_val_worker import TrainAndEvalWorker
from utils.utils import set_seed


def main():
    print("="*80)
    print("EXPERIMENT: UNet + EfficientNet-B3")
    print("="*80)
    print()
    print("Baseline (B4): 0.6448 Test Dice")
    print("Using same frozen CV splits and configuration")
    print()
    print("="*80)
    print()
    
    # Initialize config (baseline settings)
    config = Config()
    config.model_name = "unet"
    config.encoder_name = "efficientnet-b3"  # <<< ONLY CHANGE
    config.encoder_weights = "imagenet"
    config.resolution = 512
    config.batch_size = 8
    config.num_epochs = 50
    config.learning_rate = 1e-4
    
    # Override checkpoint dir for this experiment
    config.checkpoint_dir = os.path.join(config.output_dir, "checkpoints", "efficientnet_b3")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Set seed
    set_seed(config.random_state)
    
    # Create data factory
    print("Loading dataset...")
    data_factory = DataFactory(config)
    
    # Create metadata
    train_df, test_df = data_factory.create_metadata_dataframe()
    print(f"✓ Training images: {len(train_df)}")
    print(f"✓ Test images: {len(test_df)}")
    print()
    
    # Load frozen splits
    splits_path = Path("outputs/cv_splits.json")
    if not splits_path.exists():
        print(f"ERROR: Frozen splits not found at {splits_path}")
        return 1
    
    with open(splits_path) as f:
        cv_data = json.load(f)
    
    print(f"✓ Loaded frozen splits (hash: {cv_data['metadata']['train_data_hash'][:16]}...)")
    print()
    
    # Initialize trainer
    trainer = TrainAndEvalWorker(config)
    
    # Storage for results
    fold_results = []
    saved_model_paths = []
    
    # Run 5-fold cross-validation
    for fold_idx in range(5):
        print("\n" + "="*80)
        print(f"FOLD {fold_idx + 1}/5")
        print("="*80)
        
        # Get fold data
        fold_data = cv_data['folds'][fold_idx]
        train_indices = fold_data['train_indices']
        val_indices = fold_data['val_indices']
        
        print(f"Train samples: {len(train_indices)}")
        print(f"Val samples: {len(val_indices)}")
        
        # Create datasets for this fold
        train_fold_df = train_df.iloc[train_indices].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_indices].reset_index(drop=True)
        
        train_dataset = ROPDataset(
            dataframe=train_fold_df,
            config=config,
            is_train=True
        )
        
        val_dataset = ROPDataset(
            dataframe=val_fold_df,
            config=config,
            is_train=False
        )
        
        # Train fold
        model, history = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            fold=fold_idx + 1
        )
        
        # Extract best results
        best_dice = max(history['val_dice'])
        best_epoch = history['val_dice'].index(best_dice) + 1
        
        # Get path to saved model for this fold
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
        print(f"  Model saved: {model_path}")
    
    # Calculate CV statistics
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS")
    print("="*80)
    
    mean_dice = sum(f['best_val_dice'] for f in fold_results) / len(fold_results)
    std_dice = (sum((f['best_val_dice'] - mean_dice)**2 for f in fold_results) / len(fold_results))**0.5
    
    print(f"\nCross-Validation Results:")
    print(f"  Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"\nPer-fold breakdown:")
    for result in fold_results:
        print(f"  Fold {result['fold']}: {result['best_val_dice']:.4f} (epoch {result['best_epoch']})")
    
    print(f"\nBaseline (B4) CV: 0.5444")
    print(f"This (B3) CV:     {mean_dice:.4f}")
    diff_cv = mean_dice - 0.5444
    print(f"Difference:       {diff_cv:+.4f}")
    
    # ==============================================================================
    # FINAL TEST SET EVALUATION (Ensemble + TTA)
    # ==============================================================================
    print("\n" + "="*80)
    print("FINAL TEST SET EVALUATION (Ensemble + TTA)")
    print("="*80)
    print()
    print(f"Using ensemble of {len(saved_model_paths)} folds + TTA...")
    print(f"TTA transforms: {len(config.tta_transforms)}")
    print()
    
    # Create test dataset
    test_dataset = ROPDataset(
        dataframe=test_df,
        config=config,
        is_train=False
    )
    
    # Evaluate with ensemble + TTA
    test_results = trainer.evaluate_ensemble(
        test_dataset=test_dataset,
        model_paths=saved_model_paths,
        use_tta=True
    )
    
    # Display test results
    print("\n=== Test Results (Ensemble + TTA) ===")
    print(f"Mean Dice: {test_results['mean_dice']:.4f}")
    print(f"Mean IoU: {test_results['mean_iou']:.4f}")
    print(f"Exudates - Dice: {test_results['dice_exudates']:.4f}, IoU: {test_results['iou_exudates']:.4f}")
    print(f"Haemorrhages - Dice: {test_results['dice_haemorrhages']:.4f}, IoU: {test_results['iou_haemorrhages']:.4f}")
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"Baseline (B4) Test:  0.6448")
    print(f"This (B3) Test:      {test_results['mean_dice']:.4f}")
    diff_test = test_results['mean_dice'] - 0.6448
    print(f"Difference:          {diff_test:+.4f}")
    
    if test_results['mean_dice'] > 0.6448:
        print("\n✅ B3 OUTPERFORMS B4!")
    elif test_results['mean_dice'] > 0.6400:
        print("\n≈ B3 COMPARABLE TO B4")
    else:
        print("\n❌ B3 UNDERPERFORMS B4")
    
    print("\n" + "="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
