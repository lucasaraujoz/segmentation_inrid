"""
Train UNet with ASPP in Decoder - 5-fold CV

Architecture: EfficientNet-B4 + UNet with ASPP modules in decoder blocks
Experiment: ASPP at decoder levels [0, 1, 2] for multi-scale context
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).parent))

from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset import ROPDataset
from models.unet_aspp_decoder import create_unet_aspp_decoder
from utils.utils import (
    AverageMeter,
    calculate_dice_score,
    calculate_iou_score,
    set_seed
)
import segmentation_models_pytorch as smp


class ASPPDecoderTrainer:
    """Trainer for UNet with ASPP decoder."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    def create_model(self):
        """Create model with ASPP decoder."""
        model = create_unet_aspp_decoder(self.config)
        return model.to(self.device)
    
    def create_loss_function(self):
        """Create combined loss function."""
        dice_loss = smp.losses.DiceLoss(mode='multilabel', from_logits=True)
        focal_loss = smp.losses.FocalLoss(
            mode='multilabel',
            alpha=self.config.focal_alpha,
            gamma=self.config.focal_gamma
        )
        
        class CombinedLoss(nn.Module):
            def __init__(self, dice_loss, focal_loss, dice_weight=0.5, focal_weight=0.5):
                super().__init__()
                self.dice_loss = dice_loss
                self.focal_loss = focal_loss
                self.dice_weight = dice_weight
                self.focal_weight = focal_weight
            
            def forward(self, pred, target):
                dice = self.dice_loss(pred, target)
                focal = self.focal_loss(pred, target)
                return self.dice_weight * dice + self.focal_weight * focal
        
        return CombinedLoss(dice_loss, focal_loss)
    
    def train_epoch(self, model, dataloader, optimizer, criterion):
        """Train for one epoch."""
        model.train()
        
        losses = AverageMeter()
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
        
        return losses.avg
    
    def validate(self, model, dataloader, criterion):
        """Validate model."""
        model.eval()
        
        losses = AverageMeter()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validation")
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Metrics
                losses.update(loss.item(), images.size(0))
                
                # Store predictions
                preds = torch.sigmoid(outputs)
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())
                
                pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
        
        # Calculate Dice and IoU
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        dice = calculate_dice_score(all_preds > 0.5, all_targets)
        iou = calculate_iou_score(all_preds > 0.5, all_targets)
        
        # Convert to scalar
        if isinstance(dice, torch.Tensor):
            dice = dice.mean().item()
        if isinstance(iou, torch.Tensor):
            iou = iou.mean().item()
        
        return losses.avg, dice, iou
    
    def train_fold(self, train_dataset, val_dataset, fold):
        """Train one fold."""
        print(f"\nTrain samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        # Model, loss, optimizer
        model = self.create_model()
        criterion = self.create_loss_function()
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs,
            eta_min=1e-6
        )
        
        # Training loop
        best_dice = 0.0
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_dice, val_iou = self.validate(model, val_loader, criterion)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_dice'].append(val_dice)
            history['val_iou'].append(val_iou)
            
            # Scheduler step
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                patience_counter = 0
                
                # Save checkpoint
                checkpoint_dir = Path('outputs/checkpoints/aspp_decoder')
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dice': best_dice,
                    'config': self.config
                }, checkpoint_dir / f'best_model_fold{fold}.pth')
                
                print(f"Saved best model with Dice: {best_dice:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            patience = getattr(self.config, 'patience', 10)  # Default 10 if not in config
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        return best_dice, history


def main():
    print("=" * 80)
    print("EXPERIMENT: UNet with ASPP Decoder")
    print("=" * 80)
    print()
    print("Architecture: EfficientNet-B4 + UNet")
    print("Decoder: ASPP modules at levels [0, 1, 2]")
    print("Loss: Dice (50%) + Focal (50%)")
    print("Expected: +2-5% from multi-scale decoder features")
    print("=" * 80)
    print()
    
    # Config
    config = Config()
    set_seed(config.random_state)
    
    # Data
    factory = DataFactory(config)
    factory.create_metadata_dataframe()
    
    # Prepare cross-validation splits (deterministic with GroupKFold + patient_ids)
    cv_data, patient_ids, _ = factory.prepare_data_for_cross_validation()
    
    # Trainer
    trainer = ASPPDecoderTrainer(config)
    
    # 5-fold CV
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_data):
        print(f"\n{'=' * 80}")
        print(f"FOLD {fold_idx + 1}/5")
        print(f"{'=' * 80}")
        
        train_df = factory.train_df.iloc[train_idx]
        val_df = factory.train_df.iloc[val_idx]
        
        train_dataset = ROPDataset(train_df, config, is_train=True)
        val_dataset = ROPDataset(val_df, config, is_train=False)
        
        best_dice, history = trainer.train_fold(train_dataset, val_dataset, fold_idx + 1)
        
        fold_results.append({
            'fold': fold_idx + 1,
            'best_dice': best_dice,
            'history': history
        })
        
        print(f"\nFold {fold_idx + 1} Best Dice: {best_dice:.4f}")
    
    # Summary CV
    print(f"\n{'=' * 80}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'=' * 80}")
    print()
    
    fold_dices = [r['best_dice'] for r in fold_results]
    mean_dice = np.mean(fold_dices)
    std_dice = np.std(fold_dices)
    
    print(f"ASPP Decoder CV: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Baseline CV: 0.5521 ± varying")
    print()
    
    improvement_cv = ((mean_dice - 0.5521) / 0.5521) * 100
    print(f"Difference: {improvement_cv:+.2f}%")
    print()
    
    print("Per-fold results:")
    for i, dice in enumerate(fold_dices):
        print(f"  Fold {i+1}: {dice:.4f}")
    
    # ========================================================================
    # TEST SET EVALUATION: Ensemble + TTA
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("TEST SET EVALUATION: Ensemble (5 folds) + TTA")
    print(f"{'=' * 80}")
    print()
    
    # Load test dataset
    test_df = factory.test_df
    test_dataset = ROPDataset(test_df, config, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Loading {len(fold_results)} models for ensemble...")
    
    # Load all fold models
    models = []
    checkpoint_dir = Path('outputs/checkpoints/aspp_decoder')
    
    for fold_idx in range(1, 6):
        model_path = checkpoint_dir / f'best_model_fold{fold_idx}.pth'
        
        model = create_unet_aspp_decoder(config)
        checkpoint = torch.load(model_path, map_location=trainer.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(trainer.device)
        model.eval()
        models.append(model)
        print(f"  ✓ Loaded fold {fold_idx} (Dice: {checkpoint['dice']:.4f})")
    
    print()
    print("Predicting with Ensemble + TTA (4 transforms)...")
    
    # Predict with ensemble + TTA
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(trainer.device)
            masks = batch['mask']
            
            batch_preds = []
            
            # Ensemble over 5 folds
            for model in models:
                # TTA transforms
                tta_preds = []
                
                # Original
                pred = torch.sigmoid(model(images))
                tta_preds.append(pred)
                
                # Horizontal flip
                pred = torch.sigmoid(model(torch.flip(images, dims=[3])))
                pred = torch.flip(pred, dims=[3])
                tta_preds.append(pred)
                
                # Vertical flip
                pred = torch.sigmoid(model(torch.flip(images, dims=[2])))
                pred = torch.flip(pred, dims=[2])
                tta_preds.append(pred)
                
                # Rotate 90
                pred = torch.sigmoid(model(torch.rot90(images, k=1, dims=[2, 3])))
                pred = torch.rot90(pred, k=-1, dims=[2, 3])
                tta_preds.append(pred)
                
                # Average TTA
                tta_avg = torch.stack(tta_preds).mean(dim=0)
                batch_preds.append(tta_avg)
            
            # Average ensemble
            ensemble_pred = torch.stack(batch_preds).mean(dim=0)
            
            all_preds.append(ensemble_pred.cpu())
            all_targets.append(masks)
    
    # Calculate metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    test_dice = calculate_dice_score(all_preds > 0.5, all_targets)
    test_iou = calculate_iou_score(all_preds > 0.5, all_targets)
    
    # Per-class metrics
    if isinstance(test_dice, torch.Tensor) and test_dice.numel() > 1:
        dice_per_class = test_dice.numpy()
        test_dice_mean = test_dice.mean().item()
    else:
        test_dice_mean = test_dice.item() if isinstance(test_dice, torch.Tensor) else test_dice
        dice_per_class = [test_dice_mean, test_dice_mean]
    
    if isinstance(test_iou, torch.Tensor):
        test_iou = test_iou.mean().item()
    
    print()
    print(f"{'=' * 80}")
    print("TEST SET RESULTS (Ensemble + TTA)")
    print(f"{'=' * 80}")
    print(f"Test Dice: {test_dice_mean:.4f}")
    print(f"Test IoU:  {test_iou:.4f}")
    print()
    print("Per-class Dice:")
    print(f"  Microaneurysms: {dice_per_class[0]:.4f}")
    print(f"  Hemorrhages:    {dice_per_class[1]:.4f}")
    print()
    print(f"Baseline Test (Ensemble + TTA): 0.6448")
    improvement_test = ((test_dice_mean - 0.6448) / 0.6448) * 100
    print(f"Difference: {improvement_test:+.2f}%")
    print(f"{'=' * 80}")
    
    # Save final results
    results_path = Path('outputs/aspp_decoder_results.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump({
            'cv': {
                'mean_dice': mean_dice,
                'std_dice': std_dice,
                'baseline_cv': 0.5521,
                'improvement_pct': improvement_cv,
                'fold_dices': fold_dices
            },
            'test': {
                'dice': test_dice_mean,
                'iou': test_iou,
                'dice_per_class': dice_per_class.tolist() if hasattr(dice_per_class, 'tolist') else dice_per_class,
                'baseline_test': 0.6448,
                'improvement_pct': improvement_test
            }
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")


if __name__ == "__main__":
    main()
