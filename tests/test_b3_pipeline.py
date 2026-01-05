#!/usr/bin/env python3
"""
Quick test for B3: Train 1 fold for 2 epochs, test save/load/eval pipeline
"""

import sys
import json
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
    print("QUICK TEST: B3 Pipeline (1 fold, 2 epochs)")
    print("="*80)
    
    # Config
    config = Config()
    config.model_name = "unet"
    config.encoder_name = "efficientnet-b3"
    config.encoder_weights = "imagenet"
    config.resolution = 512
    config.batch_size = 8
    config.num_epochs = 2  # Quick test
    config.learning_rate = 1e-4
    config.checkpoint_dir = os.path.join(config.output_dir, "checkpoints", "efficientnet_b3_test")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    set_seed(config.random_state)
    
    # Load data
    data_factory = DataFactory(config)
    train_df, test_df = data_factory.create_metadata_dataframe()
    
    # Load frozen splits
    splits_path = Path("outputs/cv_splits.json")
    with open(splits_path) as f:
        cv_data = json.load(f)
    
    # Train only fold 1
    print("\nTraining Fold 1 (2 epochs)...")
    fold_data = cv_data['folds'][0]
    train_indices = fold_data['train_indices']
    val_indices = fold_data['val_indices']
    
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
    
    trainer = TrainAndEvalWorker(config)
    
    # Train
    model, history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        fold=1
    )
    
    print(f"\n✓ Training complete. Val Dice: {history['val_dice'][-1]:.4f}")
    
    # Check checkpoint was saved
    model_path = os.path.join(config.checkpoint_dir, "best_model_fold1.pth")
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Checkpoint not found at {model_path}")
        return 1
    
    print(f"✓ Checkpoint saved: {model_path}")
    
    # Test loading
    print("\nTesting checkpoint loading...")
    test_dataset = ROPDataset(
        dataframe=test_df,
        config=config,
        is_train=False
    )
    
    # Evaluate with single model (no ensemble, to be faster)
    test_results = trainer.evaluate_ensemble(
        test_dataset=test_dataset,
        model_paths=[model_path],
        use_tta=True
    )
    
    print(f"\n✓ Evaluation complete. Mean Dice: {test_results['mean_dice']:.4f}")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED - B3 pipeline working correctly")
    print("="*80)
    print("\nSafe to run full training!")
    
    # Cleanup test checkpoint
    print(f"\nCleaning up test checkpoint...")
    os.remove(model_path)
    os.rmdir(config.checkpoint_dir)
    print("✓ Cleanup complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
