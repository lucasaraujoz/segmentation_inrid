#!/usr/bin/env python3
"""
Fine-tuning with DANN Pre-trained Encoder on IDRiD.

This script:
1. Loads the DANN-trained encoder (aligned with FGADR)
2. Fine-tunes on IDRiD train set using same setup as baseline
3. Compares results with baseline (expected: 0.6442 test, 0.5444 CV)

Usage:
    python experiments/finetune_dann_idrid.py
"""

import sys
import json
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset import ROPDataset
from train_and_val_worker import TrainAndEvalWorker
from utils.utils import set_seed
import torch


def main():
    print("="*80)
    print("DANN FINE-TUNING: UNet + DANN-Aligned Encoder")
    print("="*80)
    print()
    print("Baseline (ImageNet): CV=0.5444, Test=0.6442")
    print("This run: DANN encoder → Fine-tune on IDRiD")
    print()
    print("="*80)
    print()
    
    # Initialize config (same as baseline)
    config = Config()
    config.model_name = "unet"
    config.encoder_name = "efficientnet-b4"
    config.encoder_weights = None  # Will load DANN weights
    config.resolution = 512
    config.batch_size = 8
    config.num_epochs = 50
    config.learning_rate = 1e-4
    
    # Override checkpoint dir for DANN fine-tuning
    config.checkpoint_dir = os.path.join(config.output_dir, "checkpoints", "dann_finetune")
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
    
    # Load frozen splits (same as baseline for fair comparison)
    splits_path = Path("outputs/cv_splits.json")
    if not splits_path.exists():
        print(f"ERROR: Frozen splits not found at {splits_path}")
        print("Run verify_baseline.py first to generate splits")
        return 1
    
    with open(splits_path) as f:
        cv_data = json.load(f)
    
    print(f"✓ Loaded frozen splits (hash: {cv_data['metadata']['train_data_hash'][:16]}...)")
    print()
    
    # Check DANN encoder exists
    dann_encoder_path = Path("outputs/dann_alignment/dann_encoder.pth")
    if not dann_encoder_path.exists():
        print(f"ERROR: DANN encoder not found at {dann_encoder_path}")
        print("Run train_dann_alignment.py first")
        return 1
    
    print(f"✓ DANN encoder found: {dann_encoder_path}")
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
        
        # Train fold with DANN encoder initialization
        print("\nInitializing model with DANN encoder...")
        
        # Temporarily modify trainer to load DANN encoder
        original_create_model = trainer.create_model
        
        def create_model_with_dann():
            """Create model and load DANN encoder weights."""
            model = original_create_model()
            
            # Load DANN encoder weights
            print(f"  Loading DANN encoder from {dann_encoder_path}")
            dann_weights = torch.load(dann_encoder_path, map_location=trainer.device)
            model.encoder.load_state_dict(dann_weights)
            print("  ✓ DANN encoder loaded successfully")
            
            return model
        
        # Replace create_model temporarily
        trainer.create_model = create_model_with_dann
        
        # Train with DANN-initialized encoder
        model, history = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            fold=fold_idx + 1
        )
        
        # Restore original create_model
        trainer.create_model = original_create_model
        
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
    
    # Calculate overall statistics
    print("\n" + "="*80)
    print("DANN FINE-TUNING RESULTS")
    print("="*80)
    
    mean_dice = sum(f['best_val_dice'] for f in fold_results) / len(fold_results)
    std_dice = (sum((f['best_val_dice'] - mean_dice)**2 for f in fold_results) / len(fold_results))**0.5
    
    print(f"\nOverall Results:")
    print(f"  Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"\nPer-fold breakdown:")
    for result in fold_results:
        print(f"  Fold {result['fold']}: {result['best_val_dice']:.4f} (epoch {result['best_epoch']})")
    
    print(f"\n{'='*80}")
    print("COMPARISON (Cross-Validation)")
    print("="*80)
    print(f"Baseline (ImageNet):  0.5444")
    print(f"DANN Fine-tuning:     {mean_dice:.4f}")
    
    improvement = ((mean_dice - 0.5444) / 0.5444) * 100
    print(f"Improvement: {improvement:+.2f}%")
    
    if improvement > 0:
        print(f"\n✅ DANN IMPROVED CV by {improvement:.2f}%")
    else:
        print(f"\n⚠️  DANN decreased CV by {abs(improvement):.2f}%")
    
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
    print(f"Mean Dice: {test_results['dice']:.4f}")
    print(f"Mean IoU: {test_results['iou']:.4f}")
    
    if 'per_class_dice' in test_results:
        print(f"Exudates - Dice: {test_results['per_class_dice'][0]:.4f}, IoU: {test_results['per_class_iou'][0]:.4f}")
        print(f"Haemorrhages - Dice: {test_results['per_class_dice'][1]:.4f}, IoU: {test_results['per_class_iou'][1]:.4f}")
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON (Test Set)")
    print("="*80)
    baseline_test = 0.6442
    dann_test = test_results['dice']
    
    print(f"Baseline (ImageNet):  {baseline_test:.4f}")
    print(f"DANN Fine-tuning:     {dann_test:.4f}")
    
    test_improvement = ((dann_test - baseline_test) / baseline_test) * 100
    print(f"Improvement: {test_improvement:+.2f}%")
    
    if test_improvement > 0:
        print(f"\n✅ DANN IMPROVED TEST by {test_improvement:.2f}%")
    else:
        print(f"\n⚠️  DANN decreased test by {abs(test_improvement):.2f}%")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"CV Improvement:   {improvement:+.2f}%")
    print(f"Test Improvement: {test_improvement:+.2f}%")
    
    if improvement > 0 and test_improvement > 0:
        print("\n🎉 DANN domain alignment helped on both CV and Test!")
    elif improvement > 0 or test_improvement > 0:
        print("\n✅ DANN showed improvement on one metric")
    else:
        print("\n⚠️  DANN did not improve over baseline")
        print("   Possible reasons:")
        print("   - Need more DANN training epochs")
        print("   - Need higher lambda_domain")
        print("   - Domain gap still too large")
    
    print("\n" + "="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
