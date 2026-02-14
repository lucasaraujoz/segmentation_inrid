"""
Pre-process EyePACS images ONCE to disk
This runs once and speeds up training dramatically
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def crop_image(image, threshold=10):
    """Remove black borders"""
    if image is None:
        return None
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Find non-black pixels
    mask = gray > threshold
    
    if mask.sum() == 0:
        return image
    
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    
    if len(rows) == 0 or len(cols) == 0:
        return image
    
    y1, y2 = rows[0], rows[-1] + 1
    x1, x2 = cols[0], cols[-1] + 1
    
    return image[y1:y2, x1:x2]


def preprocess_image(image_path, size=512):
    """Preprocess single image"""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Crop black borders
    img = crop_image(img)
    if img is None:
        return None
    
    # Apply Gaussian Blur
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Resize with aspect ratio preservation
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Pad to size x size
    pad_h = size - new_h
    pad_w = size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    img = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Convert to tensor and transpose
    img = torch.from_numpy(img).permute(2, 0, 1)  # C, H, W
    
    return img


def main():
    """Preprocess all EyePACS images"""
    source_dir = Path("/home/lucas/mestrado/tapi_inrid/eye_pacs/data/train")
    output_dir = Path("/home/lucas/mestrado/tapi_inrid/eye_pacs/data/train_preprocessed")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Source: {source_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Find all .jpg files
    jpg_files = sorted(source_dir.glob("*.jpeg"))
    if len(jpg_files) == 0:
        jpg_files = sorted(source_dir.glob("*.jpg"))
    logger.info(f"Found {len(jpg_files)} images to preprocess")
    
    if len(jpg_files) == 0:
        logger.error("No JPG files found!")
        return
    
    # Process with progress bar
    failed = 0
    for img_path in tqdm(jpg_files, desc="Preprocessing"):
        try:
            img_tensor = preprocess_image(img_path)
            
            if img_tensor is None:
                failed += 1
                continue
            
            # Save as .pt (PyTorch tensor)
            output_path = output_dir / f"{img_path.stem}.pt"
            torch.save(img_tensor, output_path)
            
        except Exception as e:
            logger.warning(f"Failed to process {img_path.name}: {e}")
            failed += 1
    
    logger.info(f"\nPreprocessing complete!")
    logger.info(f"Processed: {len(jpg_files) - failed} images")
    logger.info(f"Failed: {failed} images")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Now use: python main_phase1_v2.py --preprocess True")


if __name__ == "__main__":
    main()
