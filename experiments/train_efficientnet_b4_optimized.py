"""Train with EfficientNet-B4 - Optimized for 16GB VRAM.

Configuration:
- EfficientNet-B4 encoder (19M parameters)
- Batch size: 8 (maximize GPU usage)
- Gradient accumulation: 2 steps (effective batch = 16)
- All previous improvements (DiceLoss+Focal, Class weighting, OneCycleLR, etc.)

Expected GPU usage: ~10-12GB / 16GB available
Expected improvement: +5-8% Dice over ResNet34
"""

from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset import ROPDataset
from train_and_val_worker import TrainAndEvalWorker
from utils.utils import set_seed
import pandas as pd
import os


def main():
    """Train with optimized EfficientNet-B4 configuration."""
    
    print("=" * 80)
    print("ROP Segmentation - EfficientNet-B4 (Optimized for 16GB VRAM)")
    print("=" * 80)
    
    config = Config()
    
    # Display configuration
    print("\n--- Configuration Summary ---")
    print(f"Encoder:             {config.encoder_name}")
    print(f"Batch Size:          {config.batch_size}")
    print(f"Accumulation Steps:  {config.accumulation_steps}")
    print(f"Effective Batch:     {config.batch_size * config.accumulation_steps}")
    print(f"Epochs:              {config.num_epochs}")
    print(f"Loss:                {config.loss_type}")
    print(f"Class Weights:       {config.class_weights}")
    print(f"Scheduler:           {config.scheduler_type}")
    print(f"Early Stopping:      {config.early_stopping_patience} epochs")
    print(f"Use TTA:             {config.use_tta}")
    print(f"Use Ensemble:        {config.use_ensemble}")
    
    # Override checkpoint dir
    config.checkpoint_dir = os.path.join(config.output_dir, "checkpoints", "efficientnet_b4_optimized")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    print(f"\nCheckpoints dir:     {config.checkpoint_dir}")
    
    set_seed(config.random_state)
    
    # Initialize DataFactory
    print("\n[1/5] Initializing DataFactory...")
    data_factory = DataFactory(config)
    
    # Create metadata DataFrames
    print("\n[2/5] Creating metadata DataFrames...")
    train_df, test_df = data_factory.create_metadata_dataframe()
    
    print("\n--- Dataset Info ---")
    print(f"Training images:   {len(train_df)}")
    print(f"Test images:       {len(test_df)}")
    
    # Display class distribution
    print("\n--- Class Distribution ---")
    train_dist = data_factory.get_class_distribution(train_df)
    print("Training set:")
    print(train_dist)
    
    # Prepare cross-validation splits
    print("\n[3/5] Preparing cross-validation splits...")
    splits, patient_ids, test_df = data_factory.prepare_data_for_cross_validation()
    
    # Initialize TrainAndEvalWorker
    print("\n[4/5] Initializing TrainAndEvalWorker...")
    worker = TrainAndEvalWorker(config)
    
    # Cross-validation training
    print("\n[5/5] Starting cross-validation training...")
    print("=" * 80)
    
    fold_results = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1}/{config.n_folds}")
        print(f"{'='*80}")
        
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
        
        # Train model
        model, history = worker.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            fold=fold_idx + 1
        )
        
        # Store fold results
        best_dice = max(history['val_dice'])
        best_iou = max(history['val_iou'])
        
        fold_results.append({
            'fold': fold_idx + 1,
            'best_val_dice': best_dice,
            'best_val_iou': best_iou
        })
        
        print(f"\nFold {fold_idx + 1} completed - Best Dice: {best_dice:.4f}, Best IoU: {best_iou:.4f}")
    
    # Display cross-validation results
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 80)
    
    results_df = pd.DataFrame(fold_results)
    print("\n" + results_df.to_string(index=True))
    
    mean_dice = results_df['best_val_dice'].mean()
    std_dice = results_df['best_val_dice'].std()
    mean_iou = results_df['best_val_iou'].mean()
    std_iou = results_df['best_val_iou'].std()
    
    print(f"\nMean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Mean IoU:  {mean_iou:.4f} ± {std_iou:.4f}")
    
    # Check for unstable folds
    if std_dice > 0.10:
        print(f"\n⚠️  WARNING: High variance between folds (std={std_dice:.4f})")
        print("This may indicate training instability.")
    
    # Final evaluation on test set using ensemble + TTA
    print("\n" + "=" * 80)
    print("FINAL TEST SET EVALUATION (Ensemble + TTA)")
    print("=" * 80)
    
    # Create test dataset
    test_dataset = ROPDataset(
        dataframe=test_df,
        config=config,
        is_train=False
    )
    
    # Evaluate using ensemble of all folds with TTA
    model_paths = [
        f"{config.checkpoint_dir}/best_model_fold{i}.pth" 
        for i in range(1, config.n_folds + 1)
    ]
    
    print(f"\nUsing ensemble of {config.n_folds} folds + TTA...")
    print(f"TTA transforms: {len(config.tta_transforms)}")
    
    test_results = worker.evaluate_ensemble(
        test_dataset=test_dataset,
        model_paths=model_paths,
        use_tta=config.use_tta
    )
    
    # Compare with baseline
    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINE (ResNet34)")
    print("=" * 80)
    baseline_dice = 0.585
    improvement = ((test_results['mean_dice'] - baseline_dice) / baseline_dice) * 100
    
    print(f"\nResNet34 (Ensemble+TTA):      {baseline_dice:.4f}")
    print(f"EfficientNet-B4 (Ensemble+TTA): {test_results['mean_dice']:.4f}")
    print(f"Improvement:                    {improvement:+.2f}%")
    
    if improvement > 0:
        print(f"\n✅ EfficientNet-B4 is BETTER by {improvement:.2f}%")
    else:
        print(f"\n⚠️  EfficientNet-B4 is WORSE by {abs(improvement):.2f}%")
    
    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
