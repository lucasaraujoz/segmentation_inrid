"""
SSMD-UNet Phase 1: Quick Test (2 epochs)
For validation before full training
"""

import torch
from torch.utils.data import DataLoader, Subset
import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.ssmd_unet import SSMDUNetPhase1
from data_factory.eyepacs_dataset import EyePACSDataFactory, EyePACSDataset
from train_and_val_worker_phase1 import Phase1TrainWorker

def main():
    print("="*80)
    print("SSMD-UNet Phase 1: QUICK TEST (2 epochs)")
    print("="*80)
    
    # Configuration
    DATA_ROOT = "/home/lucas/mestrado/tapi_inrid/eye_pacs/data/train"
    CHECKPOINT_DIR = "checkpoints/phase1_test"
    
    BATCH_SIZE = 4  # Small batch for test
    NUM_EPOCHS = 2  # Only 2 epochs
    LEARNING_RATE = 0.0001
    MOMENTUM = 0.9
    IMAGE_SIZE = 512
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}\n")
    
    # Load dataset
    print("Loading EyePACS dataset...")
    factory = EyePACSDataFactory(DATA_ROOT)
    df = factory.scan_dataset()
    print(f"Total images: {len(df)}")
    
    # Use only first 100 images for test
    df_test = df.iloc[:100].reset_index(drop=True)
    print(f"Using {len(df_test)} images for test\n")
    
    # Create dataset
    dataset = EyePACSDataset(
        df_test,
        image_size=IMAGE_SIZE,
        apply_blur=True,
        blur_kernel=5,
        normalize=True,
    )
    
    # Create dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if DEVICE == "cuda" else False
    )
    print(f"Batches per epoch: {len(train_loader)}\n")
    
    # Create model
    print("Creating SSMD-UNet Phase 1...")
    model = SSMDUNetPhase1(in_channels=3, out_channels=3)
    print(f"Model created\n")
    
    # Create trainer
    trainer = Phase1TrainWorker(
        model=model,
        device=DEVICE,
        checkpoint_dir=CHECKPOINT_DIR,
    )
    
    # Train
    print("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=None,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        momentum=MOMENTUM,
        checkpoint_interval=1,
        vis_interval=1,
    )
    
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print("Ready for full training!\n")

if __name__ == "__main__":
    main()
