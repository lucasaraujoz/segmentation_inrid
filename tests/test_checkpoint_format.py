#!/usr/bin/env python3
"""Quick test: train 1 fold for 2 epochs to verify checkpoint format."""

import sys
import os
import json
import torch
import segmentation_models_pytorch as smp
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path.cwd()))

from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset import ROPDataset
from train_and_val_worker import TrainAndEvalWorker
from utils.utils import set_seed

# Import phase functions from progressive training
sys.path.append(str(Path.cwd() / "experiments"))
from train_progressive_training import train_phase_1, train_phase_2, train_phase_3


def main():
    print("="*80)
    print("QUICK TEST: Train 1 fold, 2 epochs per phase")
    print("="*80)
    print()
    
    config = Config()
    config.model_name = "unet"
    config.encoder_name = "efficientnet-b4"
    config.encoder_weights = "imagenet"
    config.resolution = 512
    config.batch_size = 4
    
    config.checkpoint_dir = os.path.join(config.output_dir, "checkpoints", "progressive_training")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    set_seed(config.random_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data_factory = DataFactory(config)
    train_df, test_df = data_factory.create_metadata_dataframe()
    
    # Load CV splits
    with open("outputs/cv_splits.json") as f:
        cv_data = json.load(f)
    
    # Train only fold 0
    fold_idx = 0
    print(f"Training Fold {fold_idx}...")
    
    fold_data = cv_data['folds'][fold_idx]
    train_indices = fold_data['train_indices']
    val_indices = fold_data['val_indices']
    
    train_fold_df = train_df.iloc[train_indices].reset_index(drop=True)
    val_fold_df = train_df.iloc[val_indices].reset_index(drop=True)
    
    train_dataset = ROPDataset(dataframe=train_fold_df, config=config, is_train=True)
    val_dataset = ROPDataset(dataframe=val_fold_df, config=config, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = smp.Unet(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=3,
        classes=2,
        activation=None
    )
    model = model.to(device)
    
    # Train 3 phases (just 2 epochs each for quick test)
    print("\n=== Testing Phase 1 (2 epochs) ===")
    # Temporarily modify to 2 epochs for quick test
    import train_progressive_training as prog
    orig_phase1 = prog.train_phase_1
    
    def quick_phase1(model, train_loader, val_loader, config, device, fold):
        # Quick version with 2 epochs
        import torch.optim as optim
        from train_progressive_training import freeze_encoder, EarlyStopping, train_epoch, validate_epoch
        
        freeze_encoder(model)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        
        dice_loss = smp.losses.DiceLoss(mode='multilabel', from_logits=True)
        focal_loss = smp.losses.FocalLoss(mode='multilabel', alpha=0.25, gamma=2.0)
        criterion = lambda pred, target: 0.5 * dice_loss(pred, target) + 0.5 * focal_loss(pred, target)
        
        best_dice = 0
        best_model_state = None
        
        for epoch in range(2):  # Quick test: 2 epochs
            print(f"Epoch {epoch+1}/2")
            train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_dice, val_iou = validate_epoch(model, val_loader, criterion, device)
            print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
            
            if val_dice > best_dice:
                best_dice = val_dice
                best_model_state = model.state_dict().copy()
        
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return model, best_dice
    
    model, phase1_dice = quick_phase1(model, train_loader, val_loader, config, device, fold_idx)
    print(f"Phase 1 Best: {phase1_dice:.4f}")
    
    # Save checkpoint (same format as baseline)
    model_path = os.path.join(config.checkpoint_dir, f"fold_{fold_idx}_best.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, model_path)
    print(f"✓ Checkpoint saved: {model_path}")
    
    # Test loading
    print("\n=== Testing checkpoint loading ===")
    trainer = TrainAndEvalWorker(config)
    test_model = trainer.create_model()
    checkpoint = torch.load(model_path, weights_only=False)
    test_model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Checkpoint loaded successfully!")
    
    # Test evaluation
    print("\n=== Testing evaluation ===")
    test_dataset = ROPDataset(dataframe=test_df, config=config, is_train=False)
    
    test_results = trainer.evaluate_ensemble(
        test_dataset=test_dataset,
        model_paths=[model_path],
        use_tta=True
    )
    
    # Results already printed
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nCheckpoint format is correct. Ready to train full experiment.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
