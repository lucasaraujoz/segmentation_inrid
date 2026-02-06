#!/usr/bin/env python3
"""
FGADR Multi-class Segmentation Experiment

Dataset: FGADR (1842 images, 2 classes)
Classes: Exudates (Hard+Soft combined) + Hemorrhage
Preprocessing: CLAHE + Bilateral Filter
Architecture: U-Net with EfficientNet-B4 encoder
Output: 2-channel multi-label segmentation with Sigmoid activation

Based on best IDRiD baseline (verify_baseline.py)
"""

import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data_factory.fgadr_multiclass_dataset import FGADRMultiClassDataset, create_fgadr_dataframe
from utils.utils import set_seed


class DiceLoss(nn.Module):
    """Dice Loss for multi-class segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) - predicted probabilities
            target: (B, C, H, W) - ground truth binary masks
        """
        pred = torch.sigmoid(pred)
        
        # Flatten spatial dimensions
        pred = pred.view(pred.size(0), pred.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)
        
        # Calculate Dice per class
        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Average across classes and batch
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined BCE + Dice Loss"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculate Dice and IoU metrics
    
    Args:
        pred: (B, C, H, W) - predicted logits
        target: (B, C, H, W) - ground truth binary masks
    """
    pred = torch.sigmoid(pred) > threshold
    pred = pred.float()
    
    # Flatten spatial dimensions
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)
    
    # Calculate per-class metrics
    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
    
    # Dice
    dice = (2.0 * intersection + 1.0) / (union + 1.0)
    
    # IoU
    iou = (intersection + 1.0) / (union - intersection + 1.0)
    
    # Average across batch and classes
    return dice.mean().item(), iou.mean().item()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_dice = 0
    total_iou = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        dice, iou = calculate_metrics(outputs.detach(), masks)
        
        # Accumulate
        total_loss += loss.item()
        total_dice += dice
        total_iou += iou
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}',
            'iou': f'{iou:.4f}'
        })
    
    return {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'iou': total_iou / num_batches
    }


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0
    total_dice = 0
    total_iou = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            dice, iou = calculate_metrics(outputs, masks)
            
            # Accumulate
            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}',
                'iou': f'{iou:.4f}'
            })
    
    return {
        'loss': total_loss / num_batches,
        'dice': total_dice / num_batches,
        'iou': total_iou / num_batches
    }


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    args = parser.parse_args()
    
    print("="*80)
    print("FGADR MULTI-CLASS SEGMENTATION")
    print("="*80)
    print()
    print("Dataset: FGADR (1842 images, 2 classes: Exudates + Hemorrhage)")
    print("Preprocessing: CLAHE + Bilateral Filter")
    print("Architecture: U-Net + EfficientNet-B4")
    print("Classes: HardExudate, SoftExudate, Hemorrhage, MA, IRMA, Neovascularization")
    print()
    print("="*80)
    print()
    
    # Configuration
    config = {
        'model_name': 'unet',
        'encoder_name': 'efficientnet-b4',
        'encoder_weights': 'imagenet',
        'num_classes': 2,  # Exudates + Hemorrhage
        'resolution': 512,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': 1e-4,
        'random_state': 42,
        'splits_file': 'experiments_fgadr/fgadr_splits.json',  # Fixed splits for reproducibility
    }
    
    # Output directory
    output_dir = Path('outputs/fgadr_multiclass')
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Set seed
    set_seed(config['random_state'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Load fixed splits
    print(f"Loading fixed splits from {config['splits_file']}...")
    splits_path = Path(config['splits_file'])
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
    print()
    
    # Create dataset
    print("Loading FGADR dataset...")
    df = create_fgadr_dataframe()
    print()
    
    # Filter dataframes by split
    train_df = df[df['filename'].isin(splits['train'])].reset_index(drop=True)
    val_df = df[df['filename'].isin(splits['val'])].reset_index(drop=True)
    test_df = df[df['filename'].isin(splits['test'])].reset_index(drop=True)
    
    # Create train/val/test datasets
    train_dataset = FGADRMultiClassDataset(
        dataframe=train_df,
        image_size=(config['resolution'], config['resolution']),
        is_train=True
    )
    
    val_dataset = FGADRMultiClassDataset(
        dataframe=val_df,
        image_size=(config['resolution'], config['resolution']),
        is_train=False
    )
    
    test_dataset = FGADRMultiClassDataset(
        dataframe=test_df,
        image_size=(config['resolution'], config['resolution']),
        is_train=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("Creating model...")
    model = smp.Unet(
        encoder_name=config['encoder_name'],
        encoder_weights=config['encoder_weights'],
        in_channels=3,
        classes=config['num_classes'],
        activation=None  # We'll use sigmoid in loss
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {num_params:,} parameters")
    print()
    
    # Loss and optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_dice = 0.0
    history = {
        'train_loss': [],
        'train_dice': [],
        'train_iou': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': [],
    }
    
    print("Starting training...")
    print("="*80)
    print()
    
    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 80)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics['dice'])
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_dice'].append(train_metrics['dice'])
        history['train_iou'].append(train_metrics['iou'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_iou'].append(val_metrics['iou'])
        
        # Print summary
        print()
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
        
        # Save best model
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_val_dice,
                'config': config
            }, checkpoint_path)
            print(f"✓ Saved best model (Dice: {best_val_dice:.4f})")
        
        print()
    
    # Save final results
    print("="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest Val Dice: {best_val_dice:.4f}")
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load(checkpoint_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate
    test_metrics = validate_epoch(model, test_loader, criterion, device)
    
    print()
    print(f"Test  - Loss: {test_metrics['loss']:.4f}, Dice: {test_metrics['dice']:.4f}, IoU: {test_metrics['iou']:.4f}")
    print("="*80)
    
    # Save test results
    test_results = {
        'test_loss': test_metrics['loss'],
        'test_dice': test_metrics['dice'],
        'test_iou': test_metrics['iou'],
        'best_val_dice': best_val_dice
    }
    
    test_results_path = output_dir / 'test_results.json'
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\n✓ Test results saved: {test_results_path}")
    
    # Save history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ History saved: {history_path}")
    
    # Save config
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved: {config_path}")
    
    print()
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
