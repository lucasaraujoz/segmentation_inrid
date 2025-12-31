"""Utility functions for the ROP segmentation project.

This module contains:
- Visualization functions
- Custom metrics
- Seed control
- Logging utilities
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import cv2


def set_seed(seed: int = 42):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visualize_sample(
    image: np.ndarray,
    mask: np.ndarray,
    prediction: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """Visualize image, ground truth mask, and optional prediction.
    
    Args:
        image: RGB image [H, W, 3] (normalized or not)
        mask: Ground truth mask [num_classes, H, W] or [H, W, num_classes]
        prediction: Optional prediction mask [num_classes, H, W]
        class_names: Names of classes
        title: Optional title for the plot
        save_path: Optional path to save the figure
    """
    # Handle mask shape
    if mask.ndim == 3:
        if mask.shape[0] < mask.shape[2]:
            # [num_classes, H, W] -> [H, W, num_classes]
            mask = np.transpose(mask, (1, 2, 0))
    
    num_classes = mask.shape[2] if mask.ndim == 3 else 1
    
    # Determine number of subplots
    if prediction is not None:
        if prediction.ndim == 3 and prediction.shape[0] < prediction.shape[2]:
            prediction = np.transpose(prediction, (1, 2, 0))
        num_cols = 2 + num_classes * 2  # image, gt, pred for each class
    else:
        num_cols = 1 + num_classes  # image, gt for each class
    
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))
    
    if num_cols == 1:
        axes = [axes]
    
    idx = 0
    
    # Plot image
    axes[idx].imshow(denormalize_image(image))
    axes[idx].set_title("Original Image")
    axes[idx].axis("off")
    idx += 1
    
    # Plot ground truth masks
    for class_idx in range(num_classes):
        class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
        
        # Ground truth
        gt_mask = mask[:, :, class_idx] if mask.ndim == 3 else mask
        axes[idx].imshow(gt_mask, cmap="gray")
        axes[idx].set_title(f"GT: {class_name}")
        axes[idx].axis("off")
        idx += 1
        
        # Prediction
        if prediction is not None:
            pred_mask = prediction[:, :, class_idx] if prediction.ndim == 3 else prediction
            axes[idx].imshow(pred_mask, cmap="gray")
            axes[idx].set_title(f"Pred: {class_name}")
            axes[idx].axis("off")
            idx += 1
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.show()


def denormalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """Denormalize image for visualization.
    
    Args:
        image: Normalized image [H, W, 3] or [3, H, W]
        mean: Mean values used for normalization
        std: Std values used for normalization
        
    Returns:
        Denormalized image [H, W, 3] in range [0, 1]
    """
    # Handle tensor
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    
    # Handle channel-first format
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # Check if image is normalized
    if image.min() < 0 or image.max() > 1:
        # Denormalize
        mean = np.array(mean)
        std = np.array(std)
        image = image * std + mean
    
    # Clip to [0, 1]
    image = np.clip(image, 0, 1)
    
    return image


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    colors: Optional[List[Tuple[int, int, int]]] = None
) -> np.ndarray:
    """Overlay segmentation mask on image.
    
    Args:
        image: RGB image [H, W, 3]
        mask: Mask [H, W] or [num_classes, H, W]
        alpha: Transparency of overlay
        colors: List of RGB colors for each class
        
    Returns:
        Image with overlay [H, W, 3]
    """
    # Denormalize image if needed
    img = denormalize_image(image)
    img = (img * 255).astype(np.uint8)
    
    # Handle mask shape
    if mask.ndim == 3:
        if mask.shape[0] < mask.shape[2]:
            # [num_classes, H, W]
            num_classes = mask.shape[0]
        else:
            # [H, W, num_classes]
            mask = np.transpose(mask, (2, 0, 1))
            num_classes = mask.shape[0]
    else:
        num_classes = 1
        mask = mask[np.newaxis, ...]
    
    # Default colors
    if colors is None:
        colors = [
            (255, 0, 0),    # Red for class 0
            (0, 255, 0),    # Green for class 1
            (0, 0, 255),    # Blue for class 2
        ]
    
    overlay = img.copy()
    
    for class_idx in range(num_classes):
        class_mask = mask[class_idx]
        if torch.is_tensor(class_mask):
            class_mask = class_mask.cpu().numpy()
        
        class_mask = (class_mask > 0.5).astype(np.uint8)
        
        color = colors[class_idx % len(colors)]
        
        # Create colored mask
        colored_mask = np.zeros_like(img)
        colored_mask[class_mask > 0] = color
        
        # Blend with image
        overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
    
    return overlay


def calculate_dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6
) -> torch.Tensor:
    """Calculate Dice score.
    
    Args:
        pred: Predicted mask [B, C, H, W]
        target: Ground truth mask [B, C, H, W]
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice score per class [C]
    """
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum(dim=(0, 2, 3))
    union = pred.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice


def calculate_iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6
) -> torch.Tensor:
    """Calculate IoU (Jaccard) score.
    
    Args:
        pred: Predicted mask [B, C, H, W]
        target: Ground truth mask [B, C, H, W]
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        IoU score per class [C]
    """
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum(dim=(0, 2, 3))
    union = pred.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3)) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_learning_rate(optimizer):
    """Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
