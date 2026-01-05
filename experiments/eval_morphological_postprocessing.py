"""
Evaluate baseline models with morphological post-processing.

Tests different post-processing configurations on test set without retraining.
Compares against baseline test Dice: 0.6448
"""

import torch
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import segmentation_models_pytorch as smp
from scipy.ndimage import binary_opening, binary_closing
from skimage.morphology import remove_small_objects
from sklearn.metrics import jaccard_score

from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset import ROPDataset
from configs.config import Config
from utils.utils import calculate_dice_score


def apply_morphological_postprocessing(
    mask: np.ndarray, 
    kernel_size: int = 3,
    min_object_size: int = 30,
    apply_opening: bool = True,
    apply_closing: bool = True
) -> np.ndarray:
    """Apply morphological operations to clean mask.
    
    Args:
        mask: Binary mask [H, W] in range [0, 1]
        kernel_size: Size of morphological kernel
        min_object_size: Minimum size of objects to keep (pixels)
        apply_opening: Whether to apply opening (removes small noise)
        apply_closing: Whether to apply closing (fills small holes)
    
    Returns:
        Cleaned binary mask [H, W]
    """
    # Convert to binary
    binary_mask = (mask > 0.5).astype(bool)
    
    # Opening: removes small isolated pixels (false positives)
    if apply_opening:
        kernel = np.ones((kernel_size, kernel_size), dtype=bool)
        binary_mask = binary_opening(binary_mask, structure=kernel)
    
    # Closing: fills small holes inside objects
    if apply_closing:
        kernel = np.ones((kernel_size, kernel_size), dtype=bool)
        binary_mask = binary_closing(binary_mask, structure=kernel)
    
    # Remove small connected components
    if min_object_size > 0:
        try:
            binary_mask = remove_small_objects(binary_mask, min_size=min_object_size)
        except:
            pass  # If no objects, skip
    
    return binary_mask.astype(np.float32)


def predict_with_tta(model, image, device):
    """Predict with test-time augmentation (4 transforms)."""
    model.eval()
    predictions = []
    
    # Original
    with torch.no_grad():
        pred = torch.sigmoid(model(image.to(device)))
        predictions.append(pred.cpu())
    
    # Horizontal flip
    with torch.no_grad():
        pred = torch.sigmoid(model(torch.flip(image.to(device), dims=[3])))
        pred = torch.flip(pred, dims=[3])
        predictions.append(pred.cpu())
    
    # Vertical flip
    with torch.no_grad():
        pred = torch.sigmoid(model(torch.flip(image.to(device), dims=[2])))
        pred = torch.flip(pred, dims=[2])
        predictions.append(pred.cpu())
    
    # Rotate 90
    with torch.no_grad():
        pred = torch.sigmoid(model(torch.rot90(image.to(device), k=1, dims=[2, 3])))
        pred = torch.rot90(pred, k=-1, dims=[2, 3])
        predictions.append(pred.cpu())
    
    # Average predictions
    return torch.stack(predictions).mean(dim=0)


def evaluate_with_postprocessing(
    models_dir: str,
    test_dataset: ROPDataset,
    config: Config,
    postprocess_config: dict
):
    """Evaluate models with morphological post-processing.
    
    Args:
        models_dir: Directory containing model checkpoints
        test_dataset: Test dataset
        config: Configuration
        postprocess_config: Dict with post-processing parameters
    
    Returns:
        Dict with results
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # Load all fold models
    models = []
    models_dir = Path(models_dir)
    for fold in range(1, 6):
        model_path = models_dir / f"best_model_fold{fold}.pth"
        if not model_path.exists():
            print(f"Warning: {model_path} not found")
            continue
        
        model = smp.Unet(
            encoder_name=config.encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=config.num_classes
        )
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Get encoder from checkpoint config
        saved_config = checkpoint.get('config', config)
        encoder_name = saved_config.encoder_name if hasattr(saved_config, 'encoder_name') else config.encoder_name
        
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=config.num_classes
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        models.append(model)
    
    print(f"Loaded {len(models)} models")
    
    # Evaluate
    all_preds_raw = []
    all_preds_postprocessed = []
    all_targets = []
    
    print("Predicting on test set...")
    for idx in tqdm(range(len(test_dataset))):
        sample = test_dataset[idx]
        image = sample['image'].unsqueeze(0)  # [1, 3, H, W]
        target = sample['mask'].numpy()  # [2, H, W]
        
        # Ensemble prediction with TTA
        ensemble_pred = []
        for model in models:
            pred = predict_with_tta(model, image, device)
            ensemble_pred.append(pred)
        
        # Average ensemble
        pred = torch.stack(ensemble_pred).mean(dim=0).squeeze(0).numpy()  # [2, H, W]
        
        # Store raw predictions
        all_preds_raw.append(pred)
        
        # Apply post-processing per class
        pred_postprocessed = np.zeros_like(pred)
        for c in range(pred.shape[0]):
            pred_postprocessed[c] = apply_morphological_postprocessing(
                pred[c],
                kernel_size=postprocess_config['kernel_size'],
                min_object_size=postprocess_config['min_object_size'],
                apply_opening=postprocess_config['apply_opening'],
                apply_closing=postprocess_config['apply_closing']
            )
        
        all_preds_postprocessed.append(pred_postprocessed)
        all_targets.append(target)
    
    # Calculate metrics
    all_preds_raw = np.array(all_preds_raw)  # [N, 2, H, W]
    all_preds_postprocessed = np.array(all_preds_postprocessed)
    all_targets = np.array(all_targets)
    
    # Dice scores (mean across classes)
    dice_raw = calculate_dice_score(
        torch.tensor(all_preds_raw > 0.5).float(),
        torch.tensor(all_targets).float()
    )
    
    dice_postprocessed = calculate_dice_score(
        torch.tensor(all_preds_postprocessed).float(),
        torch.tensor(all_targets).float()
    )
    
    # Convert to scalar (mean)
    if isinstance(dice_raw, torch.Tensor):
        dice_raw = dice_raw.mean().item()
    if isinstance(dice_postprocessed, torch.Tensor):
        dice_postprocessed = dice_postprocessed.mean().item()
    
    # Per-class Dice
    dice_per_class_raw = []
    dice_per_class_post = []
    
    for c in range(2):
        pred_c_raw = (all_preds_raw[:, c] > 0.5).reshape(-1)
        pred_c_post = all_preds_postprocessed[:, c].reshape(-1)
        target_c = all_targets[:, c].reshape(-1)
        
        # Raw
        intersection_raw = (pred_c_raw * target_c).sum()
        dice_c_raw = (2 * intersection_raw + 1e-7) / (pred_c_raw.sum() + target_c.sum() + 1e-7)
        dice_per_class_raw.append(dice_c_raw)
        
        # Post-processed
        intersection_post = (pred_c_post * target_c).sum()
        dice_c_post = (2 * intersection_post + 1e-7) / (pred_c_post.sum() + target_c.sum() + 1e-7)
        dice_per_class_post.append(dice_c_post)
    
    return {
        'dice_raw': dice_raw,
        'dice_postprocessed': dice_postprocessed,
        'dice_per_class_raw': dice_per_class_raw,
        'dice_per_class_post': dice_per_class_post,
        'improvement': dice_postprocessed - dice_raw,
        'improvement_pct': ((dice_postprocessed - dice_raw) / dice_raw) * 100
    }


if __name__ == "__main__":
    print("=" * 80)
    print("MORPHOLOGICAL POST-PROCESSING EVALUATION")
    print("=" * 80)
    print()
    print("Testing different post-processing configurations on baseline models")
    print("Baseline Test Dice (Ensemble + TTA): 0.6448")
    print()
    
    # Setup
    config = Config()
    factory = DataFactory(config)
    factory.create_metadata_dataframe()
    
    test_df = factory.test_df
    test_dataset = ROPDataset(test_df, config, is_train=False)
    
    print(f"Test set size: {len(test_dataset)}")
    print()
    
    # Test different configurations
    configs_to_test = [
        {
            'name': 'Baseline (no post-processing)',
            'kernel_size': 0,
            'min_object_size': 0,
            'apply_opening': False,
            'apply_closing': False
        },
        {
            'name': 'Opening only (kernel=3, min_size=30)',
            'kernel_size': 3,
            'min_object_size': 30,
            'apply_opening': True,
            'apply_closing': False
        },
        {
            'name': 'Opening + Closing (kernel=3, min_size=30)',
            'kernel_size': 3,
            'min_object_size': 30,
            'apply_opening': True,
            'apply_closing': True
        },
        {
            'name': 'Aggressive (kernel=5, min_size=50)',
            'kernel_size': 5,
            'min_object_size': 50,
            'apply_opening': True,
            'apply_closing': True
        },
        {
            'name': 'Conservative (kernel=2, min_size=20)',
            'kernel_size': 2,
            'min_object_size': 20,
            'apply_opening': True,
            'apply_closing': True
        }
    ]
    
    models_dir = "outputs/checkpoints/baseline_verify"  # EfficientNet-B4 baseline (0.6448 test)
    
    results = []
    for postprocess_config in configs_to_test:
        print(f"\n{'=' * 80}")
        print(f"Testing: {postprocess_config['name']}")
        print(f"{'=' * 80}")
        
        result = evaluate_with_postprocessing(
            models_dir=models_dir,
            test_dataset=test_dataset,
            config=config,
            postprocess_config=postprocess_config
        )
        
        result['config_name'] = postprocess_config['name']
        results.append(result)
        
        print(f"\nResults:")
        print(f"  Dice (raw): {result['dice_raw']:.4f}")
        print(f"  Dice (post-processed): {result['dice_postprocessed']:.4f}")
        print(f"  Improvement: {result['improvement']:+.4f} ({result['improvement_pct']:+.2f}%)")
        print(f"  Per-class Dice:")
        print(f"    Microaneurysms - Raw: {result['dice_per_class_raw'][0]:.4f}, Post: {result['dice_per_class_post'][0]:.4f}")
        print(f"    Hemorrhages    - Raw: {result['dice_per_class_raw'][1]:.4f}, Post: {result['dice_per_class_post'][1]:.4f}")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print()
    print("Configuration                               | Dice   | vs Baseline")
    print("-" * 80)
    
    for result in results:
        dice = result['dice_postprocessed']
        improvement = result['improvement_pct']
        print(f"{result['config_name']:42} | {dice:.4f} | {improvement:+.2f}%")
    
    # Best configuration
    best_result = max(results, key=lambda x: x['dice_postprocessed'])
    print()
    print(f"Best configuration: {best_result['config_name']}")
    print(f"Test Dice: {best_result['dice_postprocessed']:.4f}")
    print(f"Improvement over baseline (0.6448): {((best_result['dice_postprocessed'] - 0.6448) / 0.6448 * 100):+.2f}%")
