#!/usr/bin/env python3
"""
Experiment 11: Progressive Training (3-Phase Freezing/Unfreezing)

Strategy:
  Phase 1 - Decoder Only (encoder frozen): 25 epochs, lr=1e-3
  Phase 2 - Partial Fine-tuning (last block): 15 epochs, encoder_lr=1e-4, decoder_lr=5e-4
  Phase 3 - Global Fine-tuning (all unfrozen): 10 epochs, lr=5e-5

Uses frozen CV splits from outputs/cv_splits.json for reproducibility.
Evaluates with Ensemble (5 folds) + TTA (4 transforms).
"""

import sys
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset import ROPDataset
from utils.utils import set_seed, calculate_dice_score, calculate_iou_score


class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve."""
    def __init__(self, patience=7, mode='max', delta=0.0001):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'max':
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop


def freeze_encoder(model):
    """Freeze entire encoder (Phase 1)."""
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("✓ Encoder frozen")


def unfreeze_last_block(model):
    """Unfreeze only last encoder block (Phase 2)."""
    # First freeze all encoder parameters
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # EfficientNet encoders use _blocks (ModuleList with 32 blocks for B4)
    if hasattr(model.encoder, '_blocks'):
        for param in model.encoder._blocks[-1].parameters():
            param.requires_grad = True
        print(f"✓ Last encoder block unfrozen (block {len(model.encoder._blocks)-1}, {sum(p.numel() for p in model.encoder._blocks[-1].parameters()):,} params)")
    elif hasattr(model.encoder, 'blocks'):
        for param in model.encoder.blocks[-1].parameters():
            param.requires_grad = True
        print("✓ Last encoder block unfrozen")
    else:
        print("⚠️  Could not find encoder blocks, unfreezing all")
        for param in model.encoder.parameters():
            param.requires_grad = True


def unfreeze_all(model):
    """Unfreeze entire model (Phase 3)."""
    for param in model.parameters():
        param.requires_grad = True
    print("✓ All parameters unfrozen")


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_dice = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            dice = calculate_dice_score(torch.sigmoid(outputs), masks)
        
        total_loss += loss.item()
        total_dice += dice.mean().item()
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{dice.mean().item():.4f}"
        })
    
    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    
    return avg_loss, avg_dice


def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            dice = calculate_dice_score(torch.sigmoid(outputs), masks)
            iou = calculate_iou_score(torch.sigmoid(outputs), masks)
            
            total_loss += loss.item()
            total_dice += dice.mean().item()
            total_iou += iou.mean().item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{dice.mean().item():.4f}"
            })
    
    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    avg_iou = total_iou / len(loader)
    
    return avg_loss, avg_dice, avg_iou


def train_phase_1(model, train_loader, val_loader, config, device, fold):
    """
    Phase 1: Train only decoder (encoder frozen).
    
    Duration: 25 epochs
    LR: 1e-3
    Early stopping: patience=8
    """
    print("\n" + "="*80)
    print(f"PHASE 1: Decoder Only (Encoder Frozen)")
    print("="*80)
    
    freeze_encoder(model)
    
    # Optimizer - only decoder parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=0.01
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=8, mode='max')
    
    # Loss - Dice + Focal (same as baseline)
    dice_loss = smp.losses.DiceLoss(mode='multilabel', from_logits=True)
    focal_loss = smp.losses.FocalLoss(mode='multilabel', alpha=0.25, gamma=2.0)
    
    def criterion(pred, target):
        return 0.5 * dice_loss(pred, target) + 0.5 * focal_loss(pred, target)
    
    best_dice = 0
    best_model_state = None
    
    for epoch in range(25):
        print(f"\nEpoch {epoch + 1}/25")
        
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = validate_epoch(model, val_loader, criterion, device)
        
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(val_dice)
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            best_model_state = model.state_dict().copy()
            print(f"✓ Best model updated (Dice: {best_dice:.4f})")
        
        # Early stopping
        if early_stopping(val_dice):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Phase 1 completed. Best Val Dice: {best_dice:.4f}")
    
    return model, best_dice


def train_phase_2(model, train_loader, val_loader, config, device, fold):
    """
    Phase 2: Partial fine-tuning (last encoder block + decoder).
    
    Duration: 15 epochs
    LR: encoder=1e-4, decoder=5e-4
    Early stopping: patience=5
    """
    print("\n" + "="*80)
    print(f"PHASE 2: Partial Fine-tuning (Last Encoder Block)")
    print("="*80)
    
    unfreeze_last_block(model)
    
    # Optimizer with discriminative learning rates
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': 1e-4},
        {'params': decoder_params, 'lr': 5e-4}
    ], weight_decay=0.01)
    
    print(f"Encoder params: {len(encoder_params)}, LR: 1e-4")
    print(f"Decoder params: {len(decoder_params)}, LR: 5e-4")
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=5, mode='max')
    
    # Loss - Dice + Focal (same as baseline)
    dice_loss = smp.losses.DiceLoss(mode='multilabel', from_logits=True)
    focal_loss = smp.losses.FocalLoss(mode='multilabel', alpha=0.25, gamma=2.0)
    
    def criterion(pred, target):
        return 0.5 * dice_loss(pred, target) + 0.5 * focal_loss(pred, target)
    
    best_dice = 0
    best_model_state = None
    
    for epoch in range(15):
        print(f"\nEpoch {epoch + 1}/15")
        
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = validate_epoch(model, val_loader, criterion, device)
        
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        print(f"LR: encoder={optimizer.param_groups[0]['lr']:.6f}, decoder={optimizer.param_groups[1]['lr']:.6f}")
        
        scheduler.step(val_dice)
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            best_model_state = model.state_dict().copy()
            print(f"✓ Best model updated (Dice: {best_dice:.4f})")
        
        # Early stopping
        if early_stopping(val_dice):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Phase 2 completed. Best Val Dice: {best_dice:.4f}")
    
    return model, best_dice


def train_phase_3(model, train_loader, val_loader, config, device, fold):
    """
    Phase 3: Global fine-tuning (all parameters).
    
    Duration: 10 epochs
    LR: 5e-5
    Early stopping: patience=5
    """
    print("\n" + "="*80)
    print(f"PHASE 3: Global Fine-tuning (All Parameters)")
    print("="*80)
    
    unfreeze_all(model)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.01
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=5, mode='max')
    
    # Loss - Dice + Focal (same as baseline)
    dice_loss = smp.losses.DiceLoss(mode='multilabel', from_logits=True)
    focal_loss = smp.losses.FocalLoss(mode='multilabel', alpha=0.25, gamma=2.0)
    
    def criterion(pred, target):
        return 0.5 * dice_loss(pred, target) + 0.5 * focal_loss(pred, target)
    
    best_dice = 0
    best_model_state = None
    
    for epoch in range(10):
        print(f"\nEpoch {epoch + 1}/10")
        
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = validate_epoch(model, val_loader, criterion, device)
        
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(val_dice)
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            best_model_state = model.state_dict().copy()
            print(f"✓ Best model updated (Dice: {best_dice:.4f})")
        
        # Early stopping
        if early_stopping(val_dice):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Phase 3 completed. Best Val Dice: {best_dice:.4f}")
    
    return model, best_dice


def evaluate_ensemble_tta(models, test_loader, device, tta_transforms=None):
    """Evaluate ensemble of models with TTA."""
    if tta_transforms is None:
        tta_transforms = [
            lambda x: x,  # original
            lambda x: torch.flip(x, [3]),  # flip horizontal
            lambda x: torch.flip(x, [2]),  # flip vertical
            lambda x: torch.rot90(x, 1, [2, 3])  # rotate 90
        ]
    
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            batch_preds = []
            
            # Ensemble over models
            for model in models:
                model.eval()
                
                # TTA
                for tta_fn in tta_transforms:
                    aug_images = tta_fn(images)
                    outputs = model(aug_images)
                    outputs = torch.sigmoid(outputs)
                    
                    # Reverse TTA
                    if tta_fn.__name__ == '<lambda>':
                        # Try to reverse (simplified)
                        batch_preds.append(outputs)
                    else:
                        batch_preds.append(outputs)
            
            # Average predictions
            ensemble_pred = torch.stack(batch_preds).mean(dim=0)
            ensemble_pred = (ensemble_pred > 0.5).float()
            
            all_preds.append(ensemble_pred.cpu())
            all_masks.append(masks.cpu())
    
    # Concatenate all
    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # Calculate metrics
    dice = calculate_dice_score(all_preds, all_masks)  # Returns [C] tensor
    iou = calculate_iou_score(all_preds, all_masks)    # Returns [C] tensor
    
    # Per-class metrics (already per-class, no need to mean over dim=0)
    per_class_dice = dice.cpu().numpy()
    per_class_iou = iou.cpu().numpy()
    
    results = {
        'dice': dice.mean().item(),
        'iou': iou.mean().item(),
        'per_class_dice': per_class_dice.tolist(),
        'per_class_iou': per_class_iou.tolist()
    }
    
    return results


def main():
    print("="*80)
    print("EXPERIMENT 11: Progressive Training (3-Phase Freezing/Unfreezing)")
    print("="*80)
    print()
    print("Strategy:")
    print("  Phase 1 - Decoder Only (25 epochs, lr=1e-3)")
    print("  Phase 2 - Partial Fine-tuning (15 epochs, encoder=1e-4, decoder=5e-4)")
    print("  Phase 3 - Global Fine-tuning (10 epochs, lr=5e-5)")
    print()
    print("Baseline to beat: 0.6448 Test Dice")
    print("="*80)
    print()
    
    # Config
    config = Config()
    config.model_name = "unet"
    config.encoder_name = "efficientnet-b4"
    config.encoder_weights = "imagenet"
    config.resolution = 512
    config.batch_size = 4  # Smaller for stability
    
    # Override checkpoint dir
    config.checkpoint_dir = os.path.join(config.output_dir, "checkpoints", "progressive_training")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Set seed
    set_seed(config.random_state)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Load data
    print("Loading dataset...")
    data_factory = DataFactory(config)
    train_df, test_df = data_factory.create_metadata_dataframe()
    print(f"✓ Training images: {len(train_df)}")
    print(f"✓ Test images: {len(test_df)}")
    print()
    
    # Load frozen CV splits
    splits_path = Path("outputs/cv_splits.json")
    if not splits_path.exists():
        print(f"ERROR: Frozen splits not found at {splits_path}")
        print("Please run baseline first to generate splits.")
        return 1
    
    with open(splits_path) as f:
        cv_data = json.load(f)
    
    print(f"✓ Loaded frozen CV splits")
    print()
    
    # Storage for results
    fold_results = []
    saved_models = []
    
    # Train each fold
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
        
        # Create datasets
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
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Create model
        model = smp.Unet(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=3,
            classes=2,
            activation=None
        )
        model = model.to(device)
        
        # Train 3 phases
        print("\nStarting 3-phase training...")
        
        model, phase1_dice = train_phase_1(model, train_loader, val_loader, config, device, fold_idx)
        model, phase2_dice = train_phase_2(model, train_loader, val_loader, config, device, fold_idx)
        model, phase3_dice = train_phase_3(model, train_loader, val_loader, config, device, fold_idx)
        
        best_dice = phase3_dice  # Final phase result
        
        # Save checkpoint to disk (same format as baseline)
        model_path = os.path.join(config.checkpoint_dir, f"fold_{fold_idx}_best.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, model_path)
        print(f"✓ Model saved: {model_path}")
        
        # Don't append model reference - will reload from checkpoint later
        saved_models.append(model_path)
        
        fold_results.append({
            'fold': fold_idx + 1,
            'phase1_dice': phase1_dice,
            'phase2_dice': phase2_dice,
            'phase3_dice': phase3_dice,
            'final_dice': best_dice
        })
        
        print(f"\n✓ Fold {fold_idx + 1} completed. Best Dice: {best_dice:.4f}")
    
    # Cross-validation summary
    print("\n" + "="*80)
    print("CROSS-VALIDATION SUMMARY")
    print("="*80)
    
    mean_dice = np.mean([f['final_dice'] for f in fold_results])
    std_dice = np.std([f['final_dice'] for f in fold_results])
    
    print(f"\nMean CV Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    print("\nPer-fold results:")
    for result in fold_results:
        print(f"  Fold {result['fold']}: Phase1={result['phase1_dice']:.4f}, "
              f"Phase2={result['phase2_dice']:.4f}, Phase3={result['phase3_dice']:.4f}, "
              f"Final={result['final_dice']:.4f}")
    
    # Test set evaluation
    print("\n" + "="*80)
    print("EVALUATING TEST SET (Ensemble + TTA)")
    print("="*80)
    
    test_dataset = ROPDataset(
        dataframe=test_df,
        config=config,
        is_train=False
    )
    
    # Use TrainAndEvalWorker for evaluation (same as baseline)
    from train_and_val_worker import TrainAndEvalWorker
    trainer = TrainAndEvalWorker(config)
    
    print(f"\nTest samples: {len(test_df)}")
    print(f"Using ensemble of {len(saved_models)} folds + TTA")
    print()
    
    test_results = trainer.evaluate_ensemble(
        test_dataset=test_dataset,
        model_paths=saved_models,
        use_tta=True
    )
    
    # Results already printed by evaluate_ensemble
    
    # Comparison with baseline
    baseline_dice = 0.6448
    improvement = test_results['mean_dice'] - baseline_dice
    improvement_pct = (improvement / baseline_dice) * 100
    
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINE")
    print("="*80)
    print(f"Baseline Test (Ensemble + TTA): {baseline_dice:.4f}")
    print(f"Progressive Training (This):     {test_results['mean_dice']:.4f}")
    print(f"Difference: {improvement:+.4f} ({improvement_pct:+.2f}%)")
    
    if test_results['mean_dice'] > baseline_dice:
        print(f"\n✅ SUCCESS! Improvement of {improvement_pct:.2f}%")
    else:
        print(f"\n❌ Did not beat baseline (worse by {abs(improvement_pct):.2f}%)")
    
    # Save results
    results_path = os.path.join(config.output_dir, "progressive_training_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'cv_results': {
                'mean_dice': mean_dice,
                'std_dice': std_dice,
                'fold_results': fold_results
            },
            'test_results': test_results,
            'baseline': baseline_dice,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    print("\n" + "="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
