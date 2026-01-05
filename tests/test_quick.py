#!/usr/bin/env python3
"""Quick test of saved models."""

import torch
import json
import sys
from pathlib import Path
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

sys.path.append(str(Path.cwd()))
from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset import ROPDataset
from utils.utils import calculate_dice_score, calculate_iou_score


def main():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading data...")
    data_factory = DataFactory(config)
    train_df, test_df = data_factory.create_metadata_dataframe()
    
    test_dataset = ROPDataset(dataframe=test_df, config=config, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    print(f"Test samples: {len(test_df)}")
    
    # Load all 5 models
    print("\nLoading models...")
    models = []
    for fold_idx in range(5):
        model = smp.Unet(
            encoder_name='efficientnet-b4',
            encoder_weights=None,
            in_channels=3,
            classes=2  # exudates and hemorrhages
        )
        checkpoint = torch.load(f'outputs/checkpoints/progressive_training/fold_{fold_idx}_best.pth')
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        models.append(model)
        print(f"  ✓ Fold {fold_idx}")
    
    # TTA transforms
    tta_transforms = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[3]),
        lambda x: torch.flip(x, dims=[2]),
        lambda x: torch.flip(x, dims=[2, 3])
    ]
    
    print(f"\nEvaluating with {len(models)} folds x {len(tta_transforms)} TTA = {len(models) * len(tta_transforms)} predictions per sample...")
    
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            batch_preds = []
            
            for model in models:
                for tta_fn in tta_transforms:
                    aug_images = tta_fn(images)
                    outputs = model(aug_images)
                    outputs = torch.sigmoid(outputs)
                    batch_preds.append(outputs)
            
            ensemble_pred = torch.stack(batch_preds).mean(dim=0)
            ensemble_pred = (ensemble_pred > 0.5).float()
            
            all_preds.append(ensemble_pred.cpu())
            all_masks.append(masks.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    dice = calculate_dice_score(all_preds, all_masks)
    iou = calculate_iou_score(all_preds, all_masks)
    
    print("\n" + "="*80)
    print("TEST RESULTS (Ensemble + TTA)")
    print("="*80)
    print(f"Test Dice: {dice.mean().item():.4f}")
    print(f"Test IoU:  {iou.mean().item():.4f}")
    print()
    print("Per-class Dice:")
    class_names = ['Exudates', 'Hemorrhages']
    for i, name in enumerate(class_names):
        print(f"  {name:20s}: {dice[i].item():.4f}")
    
    print()
    print("Per-class IoU:")
    for i, name in enumerate(class_names):
        print(f"  {name:20s}: {iou[i].item():.4f}")
    
    print()
    print("="*80)
    baseline = 0.6448
    diff = dice.mean().item() - baseline
    pct = (diff / baseline) * 100
    print(f"Baseline (Ensemble+TTA): {baseline:.4f}")
    print(f"This (Progressive):      {dice.mean().item():.4f}")
    print(f"Difference:              {diff:+.4f} ({pct:+.2f}%)")
    print("="*80)
    
    if dice.mean().item() > baseline:
        print(f"\n✅ SUCCESS! Beat baseline by {pct:.2f}%")
    else:
        print(f"\n❌ Did not beat baseline (worse by {abs(pct):.2f}%)")


if __name__ == '__main__':
    main()
