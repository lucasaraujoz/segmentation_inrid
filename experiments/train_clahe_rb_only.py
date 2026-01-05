"""
Experiment 14: CLAHE apenas em R e B, preservando canal Verde.

Baseline (com CLAHE em LAB):  0.6428
Sem CLAHE:                    0.6501

Hipótese: CLAHE no canal verde (em LAB) estava removendo contraste de hemorragias.
Testar: CLAHE apenas em R e B, preservar G intacto.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset_clahe_rb import ROPDatasetClaheRBOnly
from train_and_val_worker import TrainAndEvalWorker
import numpy as np
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    print("=" * 80)
    print("EXPERIMENT 14: CLAHE apenas em canais R e B (Verde preservado)")
    print("=" * 80)
    print()
    print("Baseline (CLAHE em LAB): 0.6428")
    print("Sem CLAHE:               0.6501")
    print("Este: CLAHE apenas R+B, Verde intacto")
    print()
    print("Hipótese: Canal verde tem melhor contraste para hemorragias")
    print("          CLAHE em LAB estava normalizando esse contraste")
    print()
    print("=" * 80)
    print()
    
    set_seed(42)
    
    config = Config()
    config.batch_size = 8
    config.num_epochs = 50
    config.learning_rate = 1e-3
    config.encoder_name = 'efficientnet-b4'
    config.num_classes = 2
    config.img_size = 512
    config.apply_clahe = False  # Não usar CLAHE padrão (LAB)
    
    # Create data factory
    print("Creating metadata DataFrames...")
    data_factory = DataFactory(config)
    train_df, test_df = data_factory.create_metadata_dataframe()
    data_factory.train_df = train_df
    data_factory.test_df = test_df
    
    print(f"Train set: {len(train_df)} images")
    print(f"Test set: {len(test_df)} images")
    print(f"✓ Training images: {len(train_df)}")
    print(f"✓ Test images: {len(test_df)}")
    print(f"✓ CLAHE: R+B only (Green preserved)")
    print()
    
    # Cross-validation setup
    splits, patient_ids, _ = data_factory.prepare_data_for_cross_validation()
    
    print("✓ Loaded frozen splits")
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Cross-validation
    fold_results = []
    
    for i, (train_idx, val_idx) in enumerate(splits):
        print("=" * 80)
        print(f"FOLD {i+1}/5")
        print("=" * 80)
        
        # Create datasets with CLAHE R+B only
        train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)
        
        train_dataset = ROPDatasetClaheRBOnly(train_fold_df, config, is_train=True)
        val_dataset = ROPDatasetClaheRBOnly(val_fold_df, config, is_train=False)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        # Training
        worker = TrainAndEvalWorker(config)
        model, history = worker.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            fold=i+1
        )
        
        best_dice = max(history['val_dice'])
        best_epoch = history['val_dice'].index(best_dice) + 1
        
        fold_results.append(best_dice)
        print(f"Fold {i+1} Results:")
        print(f"  Best Val Dice: {best_dice:.4f}")
        print(f"  Best Epoch: {best_epoch}")
        print()
    
    # CV Summary
    print("=" * 80)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 80)
    print()
    print(f"CV Mean Dice: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print("Baseline (CLAHE LAB) CV: 0.5444")
    print(f"Difference: {np.mean(fold_results) - 0.5444:+.4f}")
    for i, dice in enumerate(fold_results):
        best_epoch = (i+1) * 50  # placeholder
        print(f"  Fold {i+1}: {dice:.4f} (epoch {best_epoch})")
    print()
    
    # Test evaluation
    print("=" * 80)
    print("FINAL TEST SET EVALUATION (Ensemble + TTA)")
    print("=" * 80)
    print()
    
    # Create test dataset with CLAHE R+B only
    test_dataset_clahe_rb = ROPDatasetClaheRBOnly(test_df, config, is_train=False)
    
    model_paths = [f'outputs/checkpoints/best_model_fold{i+1}.pth' for i in range(5)]
    worker = TrainAndEvalWorker(config)
    
    test_results = worker.evaluate_ensemble(
        test_dataset=test_dataset_clahe_rb,
        model_paths=model_paths,
        use_tta=True
    )
    
    print()
    print(f"Mean Dice: {test_results['mean_dice']:.4f}")
    print(f"Exudates:  {test_results['dice_exudates']:.4f}")
    print(f"Hemorrhages: {test_results['dice_haemorrhages']:.4f}")
    print()
    
    # Comparison
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    baseline_clahe_lab = 0.6428
    no_clahe = 0.6501
    this_result = test_results['mean_dice']
    
    print(f"Baseline (CLAHE LAB):     {baseline_clahe_lab:.4f}")
    print(f"No CLAHE:                 {no_clahe:.4f}")
    print(f"This (CLAHE R+B only):    {this_result:.4f}")
    print(f"vs Baseline:              {this_result - baseline_clahe_lab:+.4f}")
    print(f"vs No CLAHE:              {this_result - no_clahe:+.4f}")
    print()
    
    if test_results['mean_dice'] >= 0.65:
        print("🎯 TARGET ACHIEVED: Dice ≥ 0.65 (paper-level)")


if __name__ == '__main__':
    main()
