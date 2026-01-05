"""
Train with MODERATE augmentation + Mild Class Weights

Strategy: Balance between baseline and extreme
- Moderate augmentation (probabilities 0.6-0.7, not 0.8-0.9)
- NO MixUp/CutMix (confuses small lesions)
- Focal gamma 2.0 (baseline value)
- Class weights [1.0, 2.0] (mild, not 3.0)
- 50 epochs with patience 10 (baseline)

Expected: +2-4% from better generalization without overfitting
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys
import json
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

sys.path.append(str(Path(__file__).parent))

from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset import ROPDataset
import segmentation_models_pytorch as smp
from utils.utils import (
    AverageMeter,
    calculate_dice_score,
    calculate_iou_score,
    set_seed
)


def get_moderate_augmentation(config, is_train=True):
    """Get MODERATE augmentation pipeline.
    
    Philosophy: More than baseline, but not extreme.
    Probabilities 0.6-0.7 (vs 0.5 baseline, 0.8-0.9 extreme).
    """
    if is_train:
        return A.Compose([
            A.Resize(
                height=config.image_size[0],
                width=config.image_size[1]
            ),
            # Geometric transforms - MODERATE
            A.HorizontalFlip(p=0.6),  # Moderate increase from 0.5
            A.VerticalFlip(p=0.6),
            A.RandomRotate90(p=0.6),
            A.ShiftScaleRotate(
                shift_limit=0.15,  # Moderate (baseline: 0.1, extreme: 0.2)
                scale_limit=0.15,  # Moderate (baseline: 0.1, extreme: 0.3)
                rotate_limit=60,   # Moderate (baseline: 45, extreme: 90)
                p=0.6
            ),
            # Elastic deformations - MODERATE
            A.OneOf([
                A.ElasticTransform(p=1, alpha=130, sigma=130 * 0.05),  # Moderate
                A.GridDistortion(p=1, num_steps=6, distort_limit=0.3),
                A.OpticalDistortion(distort_limit=1.5, p=1),
            ], p=0.5),  # Moderate (baseline: 0.3, extreme: 0.7)
            # Color/intensity - MODERATE
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,  # Moderate
                    contrast_limit=0.3,
                    p=1
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=30,
                    p=1
                ),
                A.RGBShift(
                    r_shift_limit=20,
                    g_shift_limit=20,
                    b_shift_limit=20,
                    p=1
                ),
            ], p=0.6),  # Moderate probability
            # Blur and noise - LIGHT
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1),
                A.MedianBlur(blur_limit=3, p=1),
            ], p=0.3),
            # Cutout - MODERATE
            A.CoarseDropout(
                max_holes=5,
                max_height=24,
                max_width=24,
                p=0.3
            ),
            # Normalize - same as baseline
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        # Validation - same as baseline
        return A.Compose([
            A.Resize(
                height=config.image_size[0],
                width=config.image_size[1]
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])


class ModerateAugTrainer:
    """Trainer with moderate augmentation."""
    
    def __init__(self, config, fold: int):
        self.config = config
        self.fold = fold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model - same as baseline
        self.model = smp.Unet(
            encoder_name='efficientnet-b4',
            encoder_weights='imagenet',
            in_channels=3,
            classes=2
        ).to(self.device)
        
        # Loss - BASELINE focal gamma (2.0, not 4.0)
        self.dice_loss = smp.losses.DiceLoss(mode='multilabel', from_logits=True)
        self.focal_loss = smp.losses.FocalLoss(
            mode='multilabel',
            alpha=0.25,
            gamma=2.0  # BASELINE value (not 4.0)
        )
        
        # Class weights - MILD [1.0, 2.0] (not 3.0)
        self.class_weights = torch.tensor([1.0, 2.0]).to(self.device)
        
        # Combined loss
        self.criterion = self._combined_loss
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,  # 50 epochs (baseline)
            eta_min=1e-6
        )
        
        # Tracking
        self.best_dice = 0.0
        self.patience_counter = 0
        self.patience = 10  # Baseline patience
    
    def _combined_loss(self, pred, target):
        """Combined loss with mild class weights."""
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        # Apply class weights
        focal_weighted = focal * self.class_weights.mean()
        
        return 0.5 * dice + 0.5 * focal_weighted
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Metrics
                preds = (torch.sigmoid(outputs) > 0.5).float()
                dice = calculate_dice_score(preds, masks)
                iou = calculate_iou_score(preds, masks)
                
                total_loss += loss.item()
                if isinstance(dice, torch.Tensor):
                    total_dice += dice.mean().item()
                else:
                    total_dice += dice
                if isinstance(iou, torch.Tensor):
                    total_iou += iou.mean().item()
                else:
                    total_iou += iou
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        n = len(val_loader)
        return total_loss / n, total_dice / n, total_iou / n
    
    def fit(self, train_loader, val_loader, save_dir: Path):
        """Train the model."""
        save_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = save_dir / f'fold_{self.fold}_best.pth'
        
        for epoch in range(self.config.num_epochs):
            print(f'\nEpoch {epoch + 1}/{self.config.num_epochs}')
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_dice, val_iou = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Print metrics
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}')
            print(f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'dice': val_dice,
                    'iou': val_iou
                }, best_model_path)
                print(f'Saved best model with Dice: {val_dice:.4f}')
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f'\nEarly stopping after {epoch + 1} epochs')
                break
        
        return self.best_dice


def evaluate_test_set(config, checkpoint_dir: Path):
    """Evaluate on test set with Ensemble + TTA."""
    print("\n" + "=" * 80)
    print("EVALUATING TEST SET (Ensemble + TTA)")
    print("=" * 80)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test data
    factory = DataFactory(config)
    factory.create_metadata_dataframe()
    
    test_dataset = ROPDataset(
        factory.test_df,
        config,
        is_train=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load all 5 fold models
    print("Loading 5 models for ensemble...")
    models = []
    fold_dices = []
    
    for fold in range(1, 6):
        model_path = checkpoint_dir / f'fold_{fold}_best.pth'
        if not model_path.exists():
            print(f"⚠ Warning: {model_path} not found, skipping")
            continue
        
        model = smp.Unet(
            encoder_name='efficientnet-b4',
            encoder_weights='imagenet',
            in_channels=3,
            classes=2
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        models.append(model)
        fold_dices.append(checkpoint['dice'])
        print(f"  ✓ Loaded fold {fold} (Dice: {checkpoint['dice']:.4f})")
    
    if len(models) == 0:
        print("❌ No models found!")
        return
    
    print()
    
    # TTA transforms
    tta_transforms = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[3]),
        lambda x: torch.flip(x, dims=[2]),
        lambda x: torch.rot90(x, k=1, dims=[2, 3])
    ]
    
    print(f"Predicting with Ensemble + TTA (4 transforms)...")
    
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            batch_preds = []
            
            for model in models:
                model_preds = []
                
                for transform in tta_transforms:
                    img_transformed = transform(images)
                    pred = torch.sigmoid(model(img_transformed))
                    
                    if transform == tta_transforms[1]:
                        pred = torch.flip(pred, dims=[3])
                    elif transform == tta_transforms[2]:
                        pred = torch.flip(pred, dims=[2])
                    elif transform == tta_transforms[3]:
                        pred = torch.rot90(pred, k=3, dims=[2, 3])
                    
                    model_preds.append(pred)
                
                batch_preds.append(torch.stack(model_preds).mean(dim=0))
            
            final_pred = torch.stack(batch_preds).mean(dim=0)
            final_pred = (final_pred > 0.5).float()
            
            all_preds.append(final_pred.cpu())
            all_masks.append(masks.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # Calculate metrics
    dice = calculate_dice_score(all_preds, all_masks)
    iou = calculate_iou_score(all_preds, all_masks)
    
    if isinstance(dice, torch.Tensor):
        dice = dice.mean().item()
    if isinstance(iou, torch.Tensor):
        iou = iou.mean().item()
    
    # Per-class metrics
    dice_ex = calculate_dice_score(all_preds[:, 0:1], all_masks[:, 0:1])
    dice_he = calculate_dice_score(all_preds[:, 1:2], all_masks[:, 1:2])
    
    if isinstance(dice_ex, torch.Tensor):
        dice_ex = dice_ex.mean().item()
    if isinstance(dice_he, torch.Tensor):
        dice_he = dice_he.mean().item()
    
    print()
    print("=" * 80)
    print("TEST SET RESULTS (Ensemble + TTA)")
    print("=" * 80)
    print(f"Test Dice: {dice:.4f}")
    print(f"Test IoU:  {iou:.4f}")
    print()
    print("Per-class Dice:")
    print(f"  Exudates:    {dice_ex:.4f}")
    print(f"  Hemorrhages: {dice_he:.4f}")
    print()
    
    baseline_dice = 0.6448
    diff_pct = ((dice - baseline_dice) / baseline_dice) * 100
    
    print(f"Baseline Test (Ensemble + TTA): {baseline_dice:.4f}")
    print(f"Difference: {diff_pct:+.2f}%")
    print("=" * 80)
    
    # Save results
    results = {
        'test_dice': dice,
        'test_iou': iou,
        'dice_exudates': dice_ex,
        'dice_hemorrhages': dice_he,
        'baseline_dice': baseline_dice,
        'difference_percent': diff_pct,
        'fold_dices': fold_dices,
        'cv_dice_mean': np.mean(fold_dices),
        'cv_dice_std': np.std(fold_dices),
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = Path('outputs/moderate_augmentation_results.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    return dice


def main():
    """Main training function."""
    print("=" * 80)
    print("EXPERIMENT: Moderate Augmentation + Mild Class Weights")
    print("=" * 80)
    print()
    print("Architecture: EfficientNet-B4 + UNet (same as baseline)")
    print("Augmentation: MODERATE (prob 0.6-0.7, not 0.8-0.9)")
    print("  + Cutout (CoarseDropout, p=0.3)")
    print("  NO MixUp/CutMix")
    print("Class Weights: [1.0, 2.0] (mild, not 3.0)")
    print("Focal Gamma: 2.0 (baseline, not 4.0)")
    print("Training: 50 epochs, patience 10 (baseline)")
    print("Loss: Dice (50%) + Focal (50%)")
    print("Expected: +2-4% from better generalization")
    print("=" * 80)
    print()
    
    config = Config()
    
    factory = DataFactory(config)
    factory.create_metadata_dataframe()
    
    cv_data, patient_ids, _ = factory.prepare_data_for_cross_validation()
    
    print(f"Creating metadata DataFrames...")
    print(f"Train set: {len(factory.train_df)} images")
    print(f"Test set: {len(factory.test_df)} images")
    
    for i, (train_idx, val_idx) in enumerate(cv_data):
        print(f"Fold {i + 1}: {len(train_idx)} train, {len(val_idx)} val")
    
    print()
    
    checkpoint_dir = Path('outputs/checkpoints/moderate_augmentation')
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(cv_data, 1):
        print("=" * 80)
        print(f"FOLD {fold}/5")
        print("=" * 80)
        print()
        
        train_df = factory.train_df.iloc[train_idx]
        val_df = factory.train_df.iloc[val_idx]
        
        print(f"Train samples: {len(train_df)}")
        print(f"Val samples: {len(val_df)}")
        print()
        
        train_dataset = ROPDataset(
            train_df,
            config,
            is_train=True,
            transform=get_moderate_augmentation(config, is_train=True)
        )
        val_dataset = ROPDataset(
            val_df,
            config,
            is_train=False,
            transform=get_moderate_augmentation(config, is_train=False)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        trainer = ModerateAugTrainer(config, fold)
        best_dice = trainer.fit(train_loader, val_loader, checkpoint_dir)
        
        fold_results.append(best_dice)
        
        print(f"\n✓ Fold {fold} completed. Best Dice: {best_dice:.4f}")
        print()
    
    print("=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    for i, dice in enumerate(fold_results, 1):
        print(f"Fold {i}: {dice:.4f}")
    print(f"\nMean CV Dice: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print("=" * 80)
    print()
    
    test_dice = evaluate_test_set(config, checkpoint_dir)
    
    print("\n" + "=" * 80)
    print("✓ EXPERIMENT COMPLETED!")
    print("=" * 80)


if __name__ == '__main__':
    main()
