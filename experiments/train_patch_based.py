"""Train with Patch-Based Segmentation.

Strategy:
- Extract 512x512 patches from original 4288x2848 images
- Use sliding window with overlap for complete coverage
- Train on patches to leverage full resolution
- Reconstruct full predictions during evaluation

Configuration:
- Patch size: 512x512
- Overlap: 50px (~10%)
- ResNet34 encoder (baseline)
- Batch size: 16 (smaller patches = more memory efficient)

Expected improvements:
- Better small lesion detection (full resolution)
- More training samples per image
- Reduced memory usage per batch
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset_patches import ROPDatasetPatches
from train_and_val_worker import TrainAndEvalWorker
from utils.utils import set_seed
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
import json
from sklearn.model_selection import GroupKFold


def reconstruct_from_patches(patches_pred, patch_info, img_width, img_height, 
                             patch_size=512, num_classes=2):
    """Reconstruct full image prediction from patches with averaging.
    
    Args:
        patches_pred: Predictions for all patches of an image (N_patches, C, H, W)
        patch_info: List of dicts with patch positions
        img_width: Original image width
        img_height: Original image height
        patch_size: Size of patches
        num_classes: Number of classes
        
    Returns:
        Reconstructed prediction (C, img_height, img_width)
    """
    # Initialize accumulation arrays
    full_pred = np.zeros((num_classes, img_height, img_width), dtype=np.float32)
    counts = np.zeros((img_height, img_width), dtype=np.float32)
    
    # Accumulate predictions
    for i, info in enumerate(patch_info):
        x, y = info['patch_x'], info['patch_y']
        patch = patches_pred[i]  # (C, H, W)
        
        # Add to accumulator
        full_pred[:, y:y+patch_size, x:x+patch_size] += patch
        counts[y:y+patch_size, x:x+patch_size] += 1
    
    # Average overlapping regions
    counts = np.maximum(counts, 1)  # Avoid division by zero
    full_pred = full_pred / counts[np.newaxis, :, :]
    
    return full_pred


def evaluate_on_patches(worker, test_dataset, model_path, patch_size=512):
    """Evaluate model on patch-based test set with reconstruction.
    
    Args:
        worker: TrainAndEvalWorker instance
        test_dataset: ROPDatasetPatches instance
        model_path: Path to trained model
        patch_size: Size of patches
        
    Returns:
        Dictionary with metrics
    """
    print("\n" + "="*80)
    print("Evaluating on Test Set with Patch Reconstruction")
    print("="*80)
    
    # Load model
    worker.model.load_state_dict(torch.load(model_path))
    worker.model.eval()
    
    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=worker.config.batch_size,
        shuffle=False,
        num_workers=worker.config.num_workers,
        pin_memory=True
    )
    
    # Group patches by image
    image_patches = {}
    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]
        img_idx = sample['image_idx']
        
        if img_idx not in image_patches:
            image_patches[img_idx] = {
                'patches': [],
                'info': [],
                'img_name': sample['image_name'],
                'img_width': sample['img_width'],
                'img_height': sample['img_height']
            }
    
    # Get predictions for all patches
    print("\nProcessing patches...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(worker.device)
            masks = batch['mask'].to(worker.device)
            
            outputs = worker.model(images)
            preds = torch.sigmoid(outputs)
            
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
    
    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Assign predictions to images
    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]
        img_idx = sample['image_idx']
        
        image_patches[img_idx]['patches'].append(all_predictions[idx])
        image_patches[img_idx]['info'].append({
            'patch_x': sample['patch_x'],
            'patch_y': sample['patch_y']
        })
    
    # Reconstruct and evaluate each image
    print("\nReconstructing full images...")
    from utils.utils import dice_coefficient, iou_score
    
    all_dice = []
    all_iou = []
    class_dice = {class_name: [] for class_name in worker.config.classes}
    class_iou = {class_name: [] for class_name in worker.config.classes}
    
    for img_idx, data in image_patches.items():
        # Reconstruct full prediction
        patches_pred = np.array(data['patches'])
        full_pred = reconstruct_from_patches(
            patches_pred, 
            data['info'],
            data['img_width'],
            data['img_height'],
            patch_size=patch_size,
            num_classes=len(worker.config.classes)
        )
        
        # Get ground truth (reconstruct from patches)
        # Find patches for this image in test_dataset
        img_patch_indices = [i for i, p in enumerate(test_dataset.patch_info) 
                            if p['image_idx'] == img_idx]
        patches_gt = np.array([all_targets[i] for i in img_patch_indices])
        full_gt = reconstruct_from_patches(
            patches_gt,
            data['info'],
            data['img_width'],
            data['img_height'],
            patch_size=patch_size,
            num_classes=len(worker.config.classes)
        )
        
        # Threshold predictions
        full_pred_binary = (full_pred > 0.5).astype(np.float32)
        
        # Calculate metrics for this image
        img_dice = []
        img_iou = []
        
        for class_idx, class_name in enumerate(worker.config.classes):
            pred_class = full_pred_binary[class_idx]
            gt_class = full_gt[class_idx]
            
            dice = dice_coefficient(
                torch.from_numpy(pred_class).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(gt_class).unsqueeze(0).unsqueeze(0)
            ).item()
            
            iou = iou_score(
                torch.from_numpy(pred_class).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(gt_class).unsqueeze(0).unsqueeze(0)
            ).item()
            
            img_dice.append(dice)
            img_iou.append(iou)
            class_dice[class_name].append(dice)
            class_iou[class_name].append(iou)
        
        mean_dice = np.mean(img_dice)
        mean_iou = np.mean(img_iou)
        all_dice.append(mean_dice)
        all_iou.append(mean_iou)
        
        print(f"  {data['img_name']}: Dice={mean_dice:.4f}, IoU={mean_iou:.4f}")
    
    # Calculate overall metrics
    results = {
        'mean_dice': float(np.mean(all_dice)),
        'mean_iou': float(np.mean(all_iou)),
        'std_dice': float(np.std(all_dice)),
        'std_iou': float(np.std(all_iou))
    }
    
    # Per-class metrics
    for class_name in worker.config.classes:
        results[f'{class_name}_dice'] = float(np.mean(class_dice[class_name]))
        results[f'{class_name}_iou'] = float(np.mean(class_iou[class_name]))
    
    print("\n" + "="*80)
    print("Test Set Results (with Patch Reconstruction)")
    print("="*80)
    print(f"Mean Dice: {results['mean_dice']:.4f} ± {results['std_dice']:.4f}")
    print(f"Mean IoU:  {results['mean_iou']:.4f} ± {results['std_iou']:.4f}")
    print("\nPer-class metrics:")
    for class_name in worker.config.classes:
        print(f"  {class_name:15} - Dice: {results[f'{class_name}_dice']:.4f}, IoU: {results[f'{class_name}_iou']:.4f}")
    
    return results


def main():
    """Train with patch-based segmentation."""
    
    print("=" * 80)
    print("ROP Segmentation - Patch-Based Training")
    print("=" * 80)
    
    config = Config()
    
    # Patch configuration
    PATCH_SIZE = 512
    OVERLAP = 50  # 10% overlap
    
    # Adjust config for patch-based training
    config.batch_size = 8  # Reduced for memory constraints
    config.encoder_name = 'efficientnet-b4'  # Use lighter encoder for patches
    config.image_size = (PATCH_SIZE, PATCH_SIZE)
    
    # Display configuration
    print("\n--- Configuration Summary ---")
    print(f"Patch Size:          {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Overlap:             {OVERLAP}px")
    print(f"Encoder:             {config.encoder_name}")
    print(f"Batch Size:          {config.batch_size}")
    print(f"Epochs:              {config.num_epochs}")
    print(f"Loss:                {config.loss_type}")
    print(f"Scheduler:           {config.scheduler_type}")
    
    # Override checkpoint dir
    config.checkpoint_dir = os.path.join(config.output_dir, "checkpoints", "patch_based")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    print(f"\nCheckpoints dir:     {config.checkpoint_dir}")
    
    set_seed(config.random_state)
    
    # Initialize DataFactory
    print("\n[1/5] Initializing DataFactory...")
    data_factory = DataFactory(config)
    
    # Create metadata DataFrames
    print("\n[2/5] Creating metadata DataFrames...")
    train_df, test_df = data_factory.create_metadata_dataframe()
    
    print("\n--- Dataset Info ---")
    print(f"Training images:   {len(train_df)}")
    print(f"Test images:       {len(test_df)}")
    
    # Calculate number of patches
    # Assuming 4288x2848 images with 512x512 patches and 50px overlap
    stride = PATCH_SIZE - OVERLAP
    patches_per_img = int(np.ceil((4288 - PATCH_SIZE) / stride) + 1) * int(np.ceil((2848 - PATCH_SIZE) / stride) + 1)
    print(f"\nPatches per image: ~{patches_per_img}")
    print(f"Total train patches: ~{len(train_df) * patches_per_img}")
    
    # Prepare cross-validation splits
    print("\n[3/5] Preparing cross-validation splits...")
    splits, patient_ids, _ = data_factory.prepare_data_for_cross_validation()
    
    # Initialize TrainAndEvalWorker
    print("\n[4/5] Initializing TrainAndEvalWorker...")
    worker = TrainAndEvalWorker(config)
    
    # Cross-validation training with patches
    print("\n[5/5] Starting cross-validation training with patches...")
    print("=" * 80)
    
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*80}")
        print(f"Fold {fold + 1}/{len(splits)}")
        print(f"{'='*80}")
        
        # Create fold datasets with patches
        train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Training on {len(train_fold_df)} images ({len(train_fold_df) * patches_per_img} patches)")
        print(f"Validating on {len(val_fold_df)} images ({len(val_fold_df) * patches_per_img} patches)")
        
        # Create patch datasets
        train_dataset = ROPDatasetPatches(
            dataframe=train_fold_df,
            config=config,
            patch_size=PATCH_SIZE,
            overlap=OVERLAP,
            is_train=True
        )
        
        val_dataset = ROPDatasetPatches(
            dataframe=val_fold_df,
            config=config,
            patch_size=PATCH_SIZE,
            overlap=OVERLAP,
            is_train=False
        )
        
        print(f"Actual train patches: {len(train_dataset)}")
        print(f"Actual val patches: {len(val_dataset)}")
        
        # Train fold
        fold_result = worker.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            fold=fold
        )
        
        cv_results.append(fold_result)
        
        print(f"\nFold {fold + 1} Results:")
        print(f"  Best Val Dice: {fold_result['best_val_dice']:.4f}")
        print(f"  Best Val IoU:  {fold_result['best_val_iou']:.4f}")
    
    # Compute average cross-validation results
    mean_dice = np.mean([r['best_val_dice'] for r in cv_results])
    mean_iou = np.mean([r['best_val_iou'] for r in cv_results])
    std_dice = np.std([r['best_val_dice'] for r in cv_results])
    std_iou = np.std([r['best_val_iou'] for r in cv_results])
    
    print("\n" + "="*80)
    print("Cross-Validation Results")
    print("="*80)
    print(f"Mean Validation Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Mean Validation IoU:  {mean_iou:.4f} ± {std_iou:.4f}")
    
    # Evaluate on test set with best model (from best fold)
    best_fold_idx = np.argmax([r['best_val_dice'] for r in cv_results])
    best_model_path = cv_results[best_fold_idx]['best_model_path']
    
    print(f"\nUsing best model from Fold {best_fold_idx + 1} for test evaluation")
    
    # Create test dataset with patches
    test_dataset = ROPDatasetPatches(
        dataframe=test_df,
        config=config,
        patch_size=PATCH_SIZE,
        overlap=OVERLAP,
        is_train=False
    )
    
    print(f"Test patches: {len(test_dataset)}")
    
    # Evaluate with patch reconstruction
    test_results = evaluate_on_patches(
        worker=worker,
        test_dataset=test_dataset,
        model_path=best_model_path,
        patch_size=PATCH_SIZE
    )
    
    # Save results
    results = {
        'patch_size': PATCH_SIZE,
        'overlap': OVERLAP,
        'cv_results': cv_results,
        'mean_cv_dice': float(mean_dice),
        'std_cv_dice': float(std_dice),
        'mean_cv_iou': float(mean_iou),
        'std_cv_iou': float(std_iou),
        'test_results': test_results
    }
    
    results_path = os.path.join(config.output_dir, 'patch_based_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
