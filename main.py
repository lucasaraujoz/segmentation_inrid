"""Main pipeline orchestrator for ROP segmentation.

This module:
- Instantiates configuration
- Creates DataFactory for metadata management
- Prepares cross-validation splits
- Instantiates TrainAndEvalWorker
- Executes training and evaluation pipeline

NEVER contains training logic or data loading.
Only orchestrates components.
"""

from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset import ROPDataset
from train_and_val_worker import TrainAndEvalWorker
from utils.utils import set_seed
import pandas as pd


def main():
    """Main pipeline execution."""
    
    # Load configuration
    print("=" * 80)
    print("ROP Segmentation Pipeline - Exudates & Haemorrhages")
    print("=" * 80)
    
    config = Config()
    
    # Set random seed for reproducibility
    set_seed(config.random_state)
    
    # Initialize DataFactory
    print("\n[1/5] Initializing DataFactory...")
    data_factory = DataFactory(config)
    
    # Create metadata DataFrames
    print("\n[2/5] Creating metadata DataFrames...")
    train_df, test_df = data_factory.create_metadata_dataframe()
    
    # Display class distribution
    print("\n--- Training Set Distribution ---")
    train_dist = data_factory.get_class_distribution(train_df)
    print(train_dist)
    
    print("\n--- Test Set Distribution ---")
    test_dist = data_factory.get_class_distribution(test_df)
    print(test_dist)
    
    # Prepare cross-validation splits
    print("\n[3/5] Preparing cross-validation splits...")
    splits, patient_ids, test_df = data_factory.prepare_data_for_cross_validation()
    
    # Initialize TrainAndEvalWorker
    print("\n[4/5] Initializing TrainAndEvalWorker...")
    worker = TrainAndEvalWorker(config)
    
    # Cross-validation training
    print("\n[5/5] Starting cross-validation training...")
    
    fold_results = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        print("\n" + "=" * 80)
        print(f"FOLD {fold_idx + 1}/{config.n_folds}")
        print("=" * 80)
        
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
        fold_results.append({
            'fold': fold_idx + 1,
            'best_val_dice': max(history['val_dice']),
            'best_val_iou': max(history['val_iou'])
        })
    
    # Display cross-validation results
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 80)
    
    results_df = pd.DataFrame(fold_results)
    print(results_df)
    print(f"\nMean Dice: {results_df['best_val_dice'].mean():.4f} ± {results_df['best_val_dice'].std():.4f}")
    print(f"Mean IoU: {results_df['best_val_iou'].mean():.4f} ± {results_df['best_val_iou'].std():.4f}")
    
    # Final evaluation on test set using best fold model
    print("\n" + "=" * 80)
    print("FINAL TEST SET EVALUATION")
    print("=" * 80)
    
    # Use best fold model (highest validation dice)
    best_fold = results_df.loc[results_df['best_val_dice'].idxmax(), 'fold']
    best_model_path = f"{config.checkpoint_dir}/best_model_fold{int(best_fold)}.pth"
    
    # Create test dataset
    test_dataset = ROPDataset(
        dataframe=test_df,
        config=config,
        is_train=False
    )
    
    # Evaluate
    test_results = worker.evaluate(
        test_dataset=test_dataset,
        model_path=best_model_path
    )
    
    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
