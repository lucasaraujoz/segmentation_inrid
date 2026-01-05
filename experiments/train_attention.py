#!/usr/bin/env python3
"""
Training script for UNet with Attention Gates (Experiment 3).

Compares Attention Gates vs Baseline:
- Baseline (UNet + EffNet-B4): Dice 0.6442
- Attention Gates: Adds attention gates in skip connections

Expected improvements:
- Better focus on lesions (hemorrhages, exudates)
- Suppress background/healthy tissue
- Evidence from Attention U-Net paper (3000+ citations)
"""

import sys
import torch
import json
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset import ROPDataset
from train_and_val_worker import TrainAndEvalWorker
from utils.utils import set_seed


def main():
    """Run 5-fold cross-validation with Attention Gates."""
    print("="*80)
    print("EXPERIMENT 3: UNet with Attention Gates")
    print("="*80)
    print()
    print("Configuration:")
    print("- Architecture: UNet + EfficientNet-B4 + Attention Gates")
    print("- Attention: Gating mechanism in skip connections")
    print("- Resolution: 512x512")
    print("- Cross-validation: 5-fold GroupKFold")
    print("- Baseline Dice: 0.6442 (must surpass this!)")
    print()
    print("="*80)
    print()
    
    # Initialize config (baseline settings)
    config = Config()
    config.model_name = "unet_attention"  # New model type
    config.encoder_name = "efficientnet-b4"
    config.encoder_weights = "imagenet"
    config.resolution = 512
    config.batch_size = 8
    config.num_epochs = 50
    config.learning_rate = 1e-4
    
    # Set checkpoint directory
    config.checkpoint_dir = os.path.join(config.output_dir, "checkpoints", "attention_gates_fixed")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Set seed for reproducibility
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
        print("Please run save_cv_splits.py first")
        return 1
    
    with open(splits_path) as f:
        cv_data = json.load(f)
    
    print(f"✓ Loaded frozen splits (hash: {cv_data['metadata']['train_data_hash'][:16]}...)")
    print()
    
    # Initialize trainer
    trainer = TrainAndEvalWorker(config)
    
    # Storage for results
    fold_results = []
    saved_model_paths = []  # Store model paths for ensemble
    
    # Run 5-fold cross-validation
    for fold_idx in range(5):
        print("\n" + "="*80)
        print(f"FOLD {fold_idx + 1}/5")
        print("="*80)
        
        # Get fold data from frozen splits
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
        
        # Extract best results from history
        best_dice = max(history['val_dice'])
        best_epoch = history['val_dice'].index(best_dice) + 1
        
        # Get path to saved model for this fold
        model_path = os.path.join(config.checkpoint_dir, f"best_model_fold{fold_idx + 1}.pth")
        saved_model_paths.append(model_path)
        
        fold_results_dict = {
            'fold': fold_idx + 1,
            'best_val_dice': best_dice,
            'best_epoch': best_epoch
        }
        
        fold_results.append(fold_results_dict)
        
        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  Best Val Dice: {fold_results_dict['best_val_dice']:.4f}")
        print(f"  Best Epoch: {fold_results_dict['best_epoch']}")
        
        # Early comparison with baseline
        baseline_dice = 0.6442
        improvement = ((fold_results_dict['best_val_dice'] - baseline_dice) / baseline_dice) * 100
        if improvement > 0:
            print(f"  🎯 IMPROVEMENT: +{improvement:.2f}% vs baseline!")
        else:
            print(f"  ⚠️  Below baseline: {improvement:.2f}%")
    
    # Calculate overall statistics
    print("\n" + "="*80)
    print("CROSS-VALIDATION SUMMARY")
    print("="*80)
    
    mean_dice = sum(f['best_val_dice'] for f in fold_results) / len(fold_results)
    std_dice = (sum((f['best_val_dice'] - mean_dice)**2 for f in fold_results) / len(fold_results))**0.5
    
    print(f"\nOverall Results:")
    print(f"  Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"\nPer-fold breakdown:")
    for i, result in enumerate(fold_results):
        print(f"  Fold {i+1}: {result['best_val_dice']:.4f} (epoch {result['best_epoch']})")
    
    # Compare with baseline
    baseline_dice = 0.6442
    improvement = ((mean_dice - baseline_dice) / baseline_dice) * 100
    
    print(f"\n{'='*80}")
    print("COMPARISON WITH BASELINE (Cross-Validation)")
    print("="*80)
    print(f"Baseline (UNet + EffNet-B4):    0.5444 (CV)")
    print(f"Attention Gates (FIXED):        {mean_dice:.4f} (CV)")
    cv_improvement = ((mean_dice - 0.5444) / 0.5444) * 100
    print(f"CV Improvement:                 {cv_improvement:+.2f}%")
    
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
    
    if 'dice_exudates' in test_results:
        print(f"Exudates - Dice: {test_results['dice_exudates']:.4f}, IoU: {test_results['iou_exudates']:.4f}")
        print(f"Haemorrhages - Dice: {test_results['dice_haemorrhages']:.4f}, IoU: {test_results['iou_haemorrhages']:.4f}")
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON (Test Set)")
    print("="*80)
    baseline_test = 0.6448
    test_dice = test_results['mean_dice']
    test_improvement = ((test_dice - baseline_test) / baseline_test) * 100
    
    print(f"Baseline (Ensemble+TTA):        {baseline_test:.4f}")
    print(f"Attention Gates (Ensemble+TTA): {test_dice:.4f}")
    print(f"Test Improvement:               {test_improvement:+.2f}%")
    
    if test_improvement > 0:
        print(f"\n✅ SUCCESS: Attention Gates IMPROVED baseline by {test_improvement:.2f}%!")
    else:
        print(f"\n❌ FAILED: Attention Gates {test_improvement:.2f}% below baseline")
        print("Note: Bug was fixed - if still below baseline, architecture may not be suitable for this dataset")
    
    print("\n" + "="*80)
    
    # Save results
    results_path = Path("outputs/attention_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_summary = {
        'experiment': 'attention_gates',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model': config.model_name,
            'encoder': config.encoder_name,
            'resolution': config.resolution,
            'batch_size': config.batch_size,
            'epochs': config.num_epochs,
            'learning_rate': config.learning_rate
        },
        'results': {
            'mean_dice': float(mean_dice),
            'std_dice': float(std_dice),
            'baseline_dice': baseline_dice,
            'improvement_percent': float(improvement),
            'folds': fold_results
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
