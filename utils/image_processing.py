"""
Image processing utilities for retinal image enhancement.

Includes:
- Green channel enhancement (best contrast for retinal images)
- Vessel enhancement (Frangi filter, Hessian-based)
- Morphological post-processing
- CRF refinement
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import frangi, sato, meijering
from skimage.morphology import disk, opening, closing, remove_small_objects
from typing import Tuple, Optional
import torch


def extract_green_channel(image: np.ndarray) -> np.ndarray:
    """Extract and enhance green channel from RGB image.
    
    Green channel has best contrast for retinal images because:
    - Blood vessels appear darker
    - Lesions (hemorrhages, exudates) are more visible
    - Less affected by illumination variations
    
    Args:
        image: RGB image [H, W, 3]
        
    Returns:
        Enhanced green channel [H, W]
    """
    if len(image.shape) == 3:
        green = image[:, :, 1]  # Extract green channel
    else:
        green = image
    
    # Normalize to [0, 1]
    green = green.astype(np.float32) / 255.0
    
    # Enhance contrast using CLAHE on green channel
    green_uint8 = (green * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_uint8)
    
    return enhanced.astype(np.float32) / 255.0


def enhance_vessels_frangi(image: np.ndarray, scales: Tuple[int] = (1, 2, 3, 4, 5)) -> np.ndarray:
    """Enhance blood vessels using Frangi filter.
    
    Frangi filter is a Hessian-based vessel enhancement method that:
    - Detects tubular structures (vessels)
    - Works on multiple scales
    - Returns vessel probability map
    
    Args:
        image: Grayscale image [H, W] in range [0, 1]
        scales: Scales to search for vessels
        
    Returns:
        Vessel-enhanced image [H, W]
    """
    # Apply Frangi filter
    vessel_map = frangi(
        image,
        sigmas=scales,
        black_ridges=True,  # Vessels are darker than background
        alpha=0.5,
        beta=0.5,
        gamma=15
    )
    
    # Normalize to [0, 1]
    vessel_map = np.clip(vessel_map, 0, 1)
    
    return vessel_map


def enhance_vessels_sato(image: np.ndarray, scales: Tuple[int] = (1, 2, 3)) -> np.ndarray:
    """Enhance blood vessels using Sato filter (alternative to Frangi).
    
    Args:
        image: Grayscale image [H, W] in range [0, 1]
        scales: Scales to search for vessels
        
    Returns:
        Vessel-enhanced image [H, W]
    """
    vessel_map = sato(
        image,
        sigmas=scales,
        black_ridges=True
    )
    
    vessel_map = np.clip(vessel_map, 0, 1)
    return vessel_map


def combine_green_and_vessels(image: np.ndarray, vessel_weight: float = 0.3) -> np.ndarray:
    """Combine green channel with vessel enhancement.
    
    Args:
        image: RGB image [H, W, 3]
        vessel_weight: Weight for vessel map [0, 1]
        
    Returns:
        Combined enhanced image [H, W, 3] ready for network
    """
    # Extract and enhance green channel
    green = extract_green_channel(image)
    
    # Enhance vessels
    vessels = enhance_vessels_frangi(green)
    
    # Combine: emphasize vessels
    combined = green + vessel_weight * vessels
    combined = np.clip(combined, 0, 1)
    
    # Convert back to 3-channel by repeating
    # (Network expects 3 channels) - keep as float32 [0, 1]
    combined_rgb = np.stack([combined, combined, combined], axis=-1)
    
    return combined_rgb.astype(np.float32)


def preprocess_retinal_image(image: np.ndarray, use_vessel_enhancement: bool = True) -> np.ndarray:
    """Complete preprocessing pipeline for retinal images.
    
    Args:
        image: RGB image [H, W, 3]
        use_vessel_enhancement: Whether to enhance vessels
        
    Returns:
        Preprocessed image [H, W, 3]
    """
    if use_vessel_enhancement:
        return combine_green_and_vessels(image, vessel_weight=0.3)
    else:
        # Just use green channel enhancement (keep as float32 [0, 1])
        green = extract_green_channel(image)
        green_rgb = np.stack([green, green, green], axis=-1)
        return green_rgb.astype(np.float32)


# ============================================================================
# POST-PROCESSING
# ============================================================================

def morphological_postprocess(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
    """Apply morphological operations to clean up predicted mask.
    
    Operations:
    1. Opening: Remove small noise (false positives)
    2. Closing: Fill small holes
    3. Remove small objects: Remove connected components < min_size pixels
    
    Args:
        mask: Binary mask [H, W] in range [0, 1]
        min_size: Minimum object size in pixels
        
    Returns:
        Cleaned mask [H, W]
    """
    # Threshold to binary
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Define structuring element (disk is good for circular lesions)
    selem = disk(2)  # Radius 2 pixels
    
    # Opening: removes small bright spots (false positives)
    opened = opening(binary_mask, selem)
    
    # Closing: fills small dark holes
    closed = closing(opened, selem)
    
    # Remove small connected components
    # Convert to boolean for skimage
    cleaned = remove_small_objects(closed.astype(bool), min_size=min_size)
    
    return cleaned.astype(np.float32)


def postprocess_multiclass_mask(mask: np.ndarray, min_sizes: Tuple[int, int] = (30, 20)) -> np.ndarray:
    """Post-process multi-class segmentation mask.
    
    Args:
        mask: Multi-class mask [H, W, C] with C classes
        min_sizes: Minimum object sizes for each class [exudates, hemorrhages]
        
    Returns:
        Cleaned mask [H, W, C]
    """
    num_classes = mask.shape[-1]
    cleaned_mask = np.zeros_like(mask)
    
    for c in range(num_classes):
        min_size = min_sizes[c] if c < len(min_sizes) else 50
        cleaned_mask[:, :, c] = morphological_postprocess(mask[:, :, c], min_size=min_size)
    
    return cleaned_mask


def apply_crf_refinement(image: np.ndarray, mask: np.ndarray, 
                        num_iterations: int = 5) -> np.ndarray:
    """Apply dense CRF refinement to align mask with image boundaries.
    
    Note: Requires pydensecrf library
    
    Args:
        image: Original RGB image [H, W, 3]
        mask: Probability mask [H, W, C]
        num_iterations: Number of CRF iterations
        
    Returns:
        Refined mask [H, W, C]
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        print("Warning: pydensecrf not installed. Skipping CRF refinement.")
        print("Install with: pip install git+https://github.com/lucasb-eyer/pydensecrf.git")
        return mask
    
    H, W, C = mask.shape
    
    # Create DenseCRF object
    d = dcrf.DenseCRF2D(W, H, C + 1)  # +1 for background
    
    # Add background class (1 - sum of foreground)
    background = 1.0 - np.sum(mask, axis=-1, keepdims=True)
    mask_with_bg = np.concatenate([background, mask], axis=-1)
    
    # Convert to CRF format [C+1, H, W]
    mask_crf = np.transpose(mask_with_bg, (2, 0, 1))
    
    # Set unary potentials
    unary = unary_from_softmax(mask_crf)
    d.setUnaryEnergy(unary)
    
    # Add pairwise potentials (bilateral and gaussian)
    # Bilateral: considers both spatial and color similarity
    d.addPairwiseBilateral(
        sxy=80,  # Spatial std
        srgb=13,  # Color std
        rgbim=image,
        compat=10
    )
    
    # Gaussian: only spatial
    d.addPairwiseGaussian(sxy=3, compat=3)
    
    # Inference
    Q = d.inference(num_iterations)
    Q = np.array(Q).reshape((C + 1, H, W))
    
    # Remove background and transpose back
    refined = np.transpose(Q[1:], (1, 2, 0))
    
    return refined


def postprocess_batch(images: torch.Tensor, masks: torch.Tensor,
                     use_morphology: bool = True,
                     use_crf: bool = False,
                     min_sizes: Tuple[int, int] = (30, 20)) -> torch.Tensor:
    """Apply post-processing to a batch of predictions.
    
    Args:
        images: Batch of original images [B, 3, H, W]
        masks: Batch of predicted masks [B, C, H, W]
        use_morphology: Whether to apply morphological operations
        use_crf: Whether to apply CRF refinement
        min_sizes: Minimum object sizes for morphology
        
    Returns:
        Post-processed masks [B, C, H, W]
    """
    B, C, H, W = masks.shape
    processed_masks = []
    
    for i in range(B):
        # Convert to numpy [H, W, C]
        mask_np = masks[i].permute(1, 2, 0).cpu().numpy()
        image_np = images[i].permute(1, 2, 0).cpu().numpy()
        
        # Convert image from [0, 1] to [0, 255] uint8
        image_np = (image_np * 255).astype(np.uint8)
        
        # Apply morphological post-processing
        if use_morphology:
            mask_np = postprocess_multiclass_mask(mask_np, min_sizes=min_sizes)
        
        # Apply CRF refinement
        if use_crf:
            mask_np = apply_crf_refinement(image_np, mask_np)
        
        # Convert back to tensor [C, H, W]
        mask_tensor = torch.from_numpy(mask_np).permute(2, 0, 1)
        processed_masks.append(mask_tensor)
    
    return torch.stack(processed_masks)
