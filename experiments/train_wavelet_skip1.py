"""
Experiment 15: Wavelet enhancement no primeiro skip (baseline sem CLAHE).

Baseline (sem CLAHE):     0.6501
Este: sem CLAHE + wavelet no primeiro skip

Hipótese: Adicionar informação de alta frequência (bordas/detalhes) 
          melhora detecção de microlesões.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import random
from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset import ROPDataset
from train_and_val_worker import TrainAndEvalWorker


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    print("=" * 80)
    print("EXPERIMENT 15: Wavelet Skip 1 (sem CLAHE)")
    print("=" * 80)
    print()
    print("Baseline (sem CLAHE):     0.6501")
    print("Este: sem CLAHE + wavelet no primeiro skip")
    print()
    print("Wavelet config:")
    print("  - Tipo: Haar DWT 2D (nível 1)")
    print("  - Onde: Apenas primeiro skip (maior resolução)")
    print("  - O que: LH + HL + HH (alta frequência)")
    print("  - Integração: concat(skip, LH, HL, HH) → conv 1x1")
    print()
    print("=" * 80)
    print()
    
    set_seed(42)
    
    # Config
    config = Config()
    config.batch_size = 8
    config.num_epochs = 100
    config.learning_rate = 1e-4
    config.encoder_name = 'efficientnet-b4'
    config.num_classes = 2
    config.img_size = 512
    config.apply_clahe = False  # SEM CLAHE (baseline 0.6501)
    config.model_name = 'unet_wavelet_skip1'  # Modelo customizado
    
    # Create data
    print("Creating metadata DataFrames...")
    data_factory = DataFactory(config)
    train_df, test_df = data_factory.create_metadata_dataframe()
    data_factory.train_df = train_df
    data_factory.test_df = test_df
    
    print(f"✓ Training images: {len(train_df)}")
    print(f"✓ Test images: {len(test_df)}")
    print(f"✓ CLAHE: {config.apply_clahe}")
    print(f"✓ Model: UNet + Wavelet Skip 1")
    print()
    
    # Load frozen splits
    cv_file = 'outputs/frozen_cv_splits.json'
    if not os.path.exists(cv_file):
        raise FileNotFoundError(f"Frozen CV splits not found: {cv_file}")
    
    import json
    with open(cv_file, 'r') as f:
        cv_data = json.load(f)
    
    print("✓ Loaded frozen CV splits")
    print()
    
    # Training worker
    trainer = TrainAndEvalWorker(config)
    
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
    print(f"Baseline (sem CLAHE) CV: 0.5578")
    print(f"Difference: {mean_dice - 0.5578:+.4f}")
    
    for result in fold_results:
        print(f"  Fold {result['fold']}: {result['best_val_dice']:.4f} (epoch {result['best_epoch']})")
    
    # Test evaluation
    print("\n" + "="*80)
    print("FINAL TEST SET EVALUATION (Ensemble + TTA)")
    print("="*80)
    
    test_dataset = ROPDataset(
        dataframe=test_df,
        config=config,
        is_train=False
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
    print(f"Baseline (sem CLAHE):        0.6501")
    print(f"This (wavelet skip 1):       {test_results['mean_dice']:.4f}")
    print(f"Difference:                  {test_results['mean_dice'] - 0.6501:+.4f}")
    
    if test_results['mean_dice'] >= 0.65:
        print("\n🎯 TARGET ACHIEVED: Dice ≥ 0.65 (paper-level)")
    
    if test_results['mean_dice'] > 0.6501:
        print(f"\n✅ IMPROVEMENT - Wavelet skip helped!")
        print(f"   Gain: {(test_results['mean_dice'] - 0.6501) * 100:.2f} percentage points")
    elif test_results['mean_dice'] < 0.6501:
        print(f"\n❌ Wavelet skip did not help")
    else:
        print(f"\n➡️ Same performance as baseline")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
