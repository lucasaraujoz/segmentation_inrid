"""
FGADR Experiment: Wavelet Skip 1 Enhancement

Baseado no experimento IDRiD que alcançou Dice 0.6721 (+2.2% sobre baseline).
Hipótese: Wavelet no primeiro skip captura microlesões (exudates/hemorrhages) melhor.

Config:
  - Wavelet: Haar DWT 2D no primeiro skip (maior resolução)
  - Epochs: 100 (como paper SOTA)
  - Batch: 4 (como paper SOTA)
  - Resolução: 512x512
  - Preprocessing: CLAHE + Bilateral Filter (já validado)
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime

from data_factory.fgadr_multiclass_dataset import FGADRMultiClassDataset, create_fgadr_dataframe
from models.unet_wavelet_skip1 import UnetWaveletSkip1


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss"""
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, targets):
        # BCE loss
        bce_loss = self.bce(logits, targets)
        
        # Dice loss
        probs = torch.sigmoid(logits)
        smooth = 1.0
        intersection = (probs * targets).sum(dim=(2, 3))
        dice_coeff = (2. * intersection + smooth) / (probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)
        dice_loss = 1 - dice_coeff.mean()
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calculate_metrics(preds, targets):
    """Calculate Dice and IoU per class"""
    smooth = 1.0
    preds_binary = (preds > 0.5).float()
    
    # Per class
    dice_scores = []
    iou_scores = []
    
    for c in range(preds.size(1)):
        pred_c = preds_binary[:, c]
        target_c = targets[:, c]
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        iou = (intersection + smooth) / (union - intersection + smooth)
        
        dice_scores.append(dice.item())
        iou_scores.append(iou.item())
    
    return np.mean(dice_scores), np.mean(iou_scores)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        logits = model(images)
        loss = criterion(logits, masks)
        
        loss.backward()
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            dice, iou = calculate_metrics(probs, masks)
        
        total_loss += loss.item()
        total_dice += dice
        total_iou += iou
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}',
            'iou': f'{iou:.4f}'
        })
    
    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def validate_epoch(model, loader, criterion, device):
    """Validate one epoch"""
    model.eval()
    
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            loss = criterion(logits, masks)
            
            probs = torch.sigmoid(logits)
            dice, iou = calculate_metrics(probs, masks)
            
            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}',
                'iou': f'{iou:.4f}'
            })
    
    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def main():
    print("=" * 80)
    print("FGADR EXPERIMENT: Wavelet Skip 1")
    print("=" * 80)
    print()
    print("Baseado no experimento IDRiD que alcançou Dice 0.6721")
    print()
    print("Configuração:")
    print("  - Modelo: UNet + EfficientNet-B4 + Wavelet Skip 1")
    print("  - Wavelet: Haar DWT 2D (LH + HL + HH)")
    print("  - Aplicado: Primeiro skip (maior resolução)")
    print("  - Epochs: 100")
    print("  - Batch Size: 4")
    print("  - Learning Rate: 1e-4")
    print("  - Optimizer: Adam")
    print("  - Loss: Dice (0.5) + BCE (0.5)")
    print("  - Scheduler: ReduceLROnPlateau (patience=5)")
    print()
    print("=" * 80)
    print()
    
    # Config
    config = {
        'resolution': 512,
        'batch_size': 4,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'num_classes': 2,
        'encoder_name': 'efficientnet-b4',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    device = torch.device(config['device'])
    print(f"Device: {device}")
    print()
    
    # Load splits
    splits_file = Path(__file__).parent / 'fgadr_splits.json'
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    print(f"✓ Loaded splits: {splits_file}")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val: {len(splits['val'])} images")
    print(f"  Test: {len(splits['test'])} images")
    print()
    
    # Create datasets
    df = create_fgadr_dataframe()
    
    train_df = df[df['filename'].isin(splits['train'])].reset_index(drop=True)
    val_df = df[df['filename'].isin(splits['val'])].reset_index(drop=True)
    test_df = df[df['filename'].isin(splits['test'])].reset_index(drop=True)
    
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
    
    print("✓ Datasets created")
    print()
    
    # Dataloaders
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    print("Creating model...")
    model = UnetWaveletSkip1(
        encoder_name=config['encoder_name'],
        encoder_weights='imagenet',
        in_channels=3,
        classes=config['num_classes']
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()
    
    # Loss, optimizer, scheduler
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Output directory
    output_dir = Path(__file__).parent.parent / 'outputs' / 'fgadr_wavelet_skip1'
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    print(f"✓ Output directory: {output_dir}")
    print()
    
    # Training loop
    best_val_dice = 0.0
    history = {
        'train_loss': [],
        'train_dice': [],
        'train_iou': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': [],
        'lr': []
    }
    
    print("=" * 80)
    print("Starting Training")
    print("=" * 80)
    print()
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 80)
        
        # Train
        train_loss, train_dice, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_dice, val_iou = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        history['lr'].append(current_lr)
        
        # Print summary
        print()
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
                'config': config
            }, checkpoint_path)
            print(f"✓ Saved best model (Dice: {val_dice:.4f})")
    
    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)
    print()
    
    # Load best model
    checkpoint = torch.load(checkpoint_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_dice, test_iou = validate_epoch(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Dice: {test_dice:.4f}")
    print(f"  IoU: {test_iou:.4f}")
    print(f"  Best Val Dice: {best_val_dice:.4f}")
    print()
    
    # Save results
    results = {
        'test_loss': test_loss,
        'test_dice': test_dice,
        'test_iou': test_iou,
        'best_val_dice': best_val_dice,
        'best_val_epoch': checkpoint['epoch'],
        'config': config,
        'history': history
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {output_dir / 'results.json'}")
    print()
    
    # Comparison
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"FGADR Baseline (50 epochs):  ~0.57 (esperado)")
    print(f"This (Wavelet Skip 1):       {test_dice:.4f}")
    print(f"IDRiD Wavelet Skip 1:        0.6721")
    print()
    
    if test_dice >= 0.60:
        print("🎯 TARGET: Dice ≥ 0.60 - ACHIEVED!")
    
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
