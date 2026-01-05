"""Train with ASPP in Bottleneck - Experiment 2.

This experiment adds ASPP (Atrous Spatial Pyramid Pooling) to the bottleneck:
- Baseline: UNet + EfficientNet-B4 (Dice 0.6442)
- New: UNet + EfficientNet-B4 + ASPP bottleneck

ASPP Configuration:
- Dilation rates: [6, 12, 18]
- Captures multi-scale context
- Small hemorrhages (rate 6) vs larger exudates (rate 18)

Expected improvement: +2-3% Dice
Target: 0.66-0.67 Dice
"""

from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset import ROPDataset
from train_and_val_worker import TrainAndEvalWorker
from utils.utils import set_seed
import pandas as pd
import os


def main():
    """Train with ASPP in bottleneck."""
    
    print("=" * 80)
    print("EXPERIMENT 2: Baseline + ASPP Bottleneck")
    print("=" * 80)
    
    config = Config()
    
    # Configure for ASPP experiment
    config.model_name = "unet_aspp"
    config.use_aspp = True
    config.aspp_rates = [6, 12, 18]  # Multi-scale: small, medium, large
    config.aspp_channels = 256
    
    # Keep same loss as baseline
    config.loss_type = "dice_focal"
    config.dice_weight = 0.5
    config.focal_weight = 0.5
    
    # Override checkpoint dir
    config.checkpoint_dir = os.path.join(
        config.output_dir, "checkpoints", "aspp_experiment"
    )
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Display configuration
    print("\n--- Configuration Summary ---")
    print(f"Model:               {config.model_name} ⭐ NEW!")
    print(f"Encoder:             {config.encoder_name}")
    print(f"Resolution:          {config.image_size}")
    print(f"Batch Size:          {config.batch_size}")
    print(f"Effective Batch:     {config.batch_size * config.accumulation_steps}")
    print(f"Epochs:              {config.num_epochs}")
    print(f"\nASPP Configuration:")
    print(f"  Enabled:           {config.use_aspp}")
    print(f"  Dilation rates:    {config.aspp_rates}")
    print(f"  Output channels:   {config.aspp_channels}")
    print(f"\nLoss Configuration:")
    print(f"  Type:              {config.loss_type}")
    print(f"  Dice weight:       {config.dice_weight}")
    print(f"  Focal weight:      {config.focal_weight}")
    print(f"  Class Weights:     {config.class_weights}")
    print(f"\nScheduler:           {config.scheduler_type}")
    print(f"Early Stopping:      {config.early_stopping_patience} epochs")
    print(f"Checkpoints dir:     {config.checkpoint_dir}")
    
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
    print("COMPARISON WITH BASELINE")
    print("=" * 80)
    
    baseline_dice = 0.6442
    baseline_exudates = 0.7046
    baseline_haemorrhages = 0.5837
    
    improvement = ((test_results['mean_dice'] - baseline_dice) / baseline_dice) * 100
    
    print(f"\nBaseline (UNet + EffNet-B4):    {baseline_dice:.4f}")
    print(f"  - Exudates:                    {baseline_exudates:.4f}")
    print(f"  - Haemorrhages:                {baseline_haemorrhages:.4f}")
    
    print(f"\nWith ASPP Bottleneck:           {test_results['mean_dice']:.4f}")
    print(f"  - Exudates:                    {test_results.get('exudates_dice', 'N/A')}")
    print(f"  - Haemorrhages:                {test_results.get('haemorrhages_dice', 'N/A')}")
    
    print(f"\nImprovement:                    {improvement:+.2f}%")
    
    if improvement > 0:
        print(f"\n✅ ASPP IMPROVED results by {improvement:.2f}%")
        if improvement >= 2.0:
            print(f"   🎯 Target achieved! (+2-3% expected)")
        else:
            print(f"   ⚠️  Below target (+2-3% expected)")
    else:
        print(f"\n❌ ASPP DECREASED results by {abs(improvement):.2f}%")
        print(f"   Consider adjusting ASPP rates or reverting to baseline")
    
    print("\n" + "=" * 80)
    print("Experiment 2 completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
