"""
TrainAndEvalWorker for SSMD-UNet Phase 1
Handles training, validation, and checkpoint saving
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
from pathlib import Path
import logging
from typing import Optional, Dict
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Phase1TrainWorker:
    """
    Trains SSMD-UNet Phase 1 (Autoencoder)
    - Unsupervised learning with reconstruction task
    - Saves checkpoints every N epochs
    - Generates visualization every 10 epochs
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints/phase1",
        use_amp: bool = False,  # Disabled - slower for this model
        use_compile: bool = False,  # Disabled - needs C compiler
    ):
        """
        Initialize trainer
        
        Args:
            model: SSMD-UNet Phase 1 model
            device: Device to use (cuda/cpu)
            checkpoint_dir: Directory to save checkpoints
            use_amp: Use Automatic Mixed Precision (faster, less memory)
            use_compile: Use torch.compile() (PyTorch 2.0+, 20-30% faster)
        """
        self.model = model.to(device)
        
        # Compile model for speed (PyTorch 2.0+)
        if use_compile and hasattr(torch, 'compile'):
            logger.info("🚀 Using torch.compile() for faster training")
            self.model = torch.compile(self.model, mode='reduce-overhead')
        
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        
        if use_amp:
            logger.info("⚡ Using Automatic Mixed Precision (AMP) training")
        
        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        momentum: float = 0.9,
        checkpoint_interval: int = 10,
        vis_interval: int = 10,
    ):
        """
        Train Phase 1 (unsupervised reconstruction)
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (optional)
            num_epochs: Number of epochs to train
            learning_rate: SGD learning rate (paper: 0.0001)
            momentum: SGD momentum (paper: 0.9)
            checkpoint_interval: Save checkpoint every N epochs
            vis_interval: Generate visualizations every N epochs
        """
        # Optimizer and loss
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum
        )
        criterion = nn.MSELoss()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting Phase 1 training for {num_epochs} epochs")
        logger.info(f"Batch size: {train_loader.batch_size}")
        logger.info(f"Total images: {len(train_loader.dataset)}")
        logger.info(f"LR: {learning_rate}, Momentum: {momentum}")
        logger.info(f"{'='*80}\n")
        
        # Track times for ETA
        epoch_times = []
        
        # Main epoch loop with progress bar
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train epoch
            train_loss = self._train_epoch(train_loader, optimizer, criterion, epoch, num_epochs)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader, criterion)
                self.val_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            # Calculate ETA
            avg_epoch_time = np.mean(epoch_times[-10:])  # Average of last 10 epochs
            remaining_epochs = num_epochs - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_hours = eta_seconds / 3600
            
            # Formatted logging
            log_msg = f"[Epoch {epoch+1:3d}/{num_epochs}] "
            log_msg += f"Loss: {train_loss:.6f} | "
            log_msg += f"Time: {epoch_time:.1f}s | "
            log_msg += f"ETA: {eta_hours:.1f}h"
            
            if val_loss is not None:
                log_msg += f" | Val: {val_loss:.6f}"
            
            logger.info(log_msg)
            
            # Checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                self._save_checkpoint(epoch + 1)
                logger.info(f"✓ Checkpoint saved at epoch {epoch+1}")
            
            # Visualization
            if (epoch + 1) % vis_interval == 0:
                self._visualize_reconstruction(val_loader or train_loader, epoch + 1)
                logger.info(f"✓ Visualization saved at epoch {epoch+1}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Training completed! Total time: {sum(epoch_times)/3600:.1f}h")
        logger.info(f"{'='*80}\n")
        self._save_checkpoint(num_epochs, is_final=True)
    
    def _train_epoch(self, train_loader: DataLoader, optimizer, criterion, epoch, total_epochs) -> float:
        """Single training epoch with progress bar"""
        self.model.train()
        total_loss = 0.0
        
        # Progress bar for batches
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{total_epochs}",
            leave=True,
            ncols=100
        )
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    reconstruction = self.model(images)
                    loss = criterion(reconstruction, images)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Regular forward/backward
                reconstruction = self.model(images)
                loss = criterion(reconstruction, images)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
        
        pbar.close()
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader, criterion) -> float:
        """Validation epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                reconstruction = self.model(images)
                loss = criterion(reconstruction, images)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _save_checkpoint(self, epoch: int, is_final: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'encoder_state': self.model.encoder.state_dict(),
            'decoder_rec_state': self.model.decoder_rec.state_dict(),
            'model_state': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        if is_final:
            encoder_path = self.checkpoint_dir / "encoder_final.pth"
            decoder_path = self.checkpoint_dir / "decoder_rec_final.pth"
            checkpoint_path = self.checkpoint_dir / "phase1_final.pth"
        else:
            encoder_path = self.checkpoint_dir / f"encoder_epoch_{epoch}.pth"
            decoder_path = self.checkpoint_dir / f"decoder_rec_epoch_{epoch}.pth"
            checkpoint_path = self.checkpoint_dir / f"phase1_epoch_{epoch}.pth"
        
        # Save individual components
        torch.save(checkpoint['encoder_state'], encoder_path)
        torch.save(checkpoint['decoder_rec_state'], decoder_path)
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.current_epoch = checkpoint.get('epoch', 0)
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def _visualize_reconstruction(self, data_loader: DataLoader, epoch: int):
        """
        Generate visualization of reconstructions every N epochs
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get first batch
            batch = next(iter(data_loader))
            images = batch['image'][:4].to(self.device)  # First 4 images
            
            # Reconstruct
            reconstructions, _, _ = self.model(images)
            
            # Move to CPU and convert to numpy
            images_np = images.cpu().numpy()
            recon_np = reconstructions.cpu().numpy()
            
            # Create visualization
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            for i in range(4):
                # Original
                img_orig = np.transpose(images_np[i], (1, 2, 0))
                img_orig = np.clip(img_orig, 0, 1)
                axes[0, i].imshow(img_orig)
                axes[0, i].set_title(f"Original {i+1}")
                axes[0, i].axis('off')
                
                # Reconstruction
                img_recon = np.transpose(recon_np[i], (1, 2, 0))
                img_recon = np.clip(img_recon, 0, 1)
                axes[1, i].imshow(img_recon)
                axes[1, i].set_title(f"Reconstructed {i+1}")
                axes[1, i].axis('off')
            
            plt.tight_layout()
            
            # Save figure
            vis_dir = self.checkpoint_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)
            vis_path = vis_dir / f"epoch_{epoch:03d}.png"
            plt.savefig(vis_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved: {vis_path}")
            
            # Also compute MSE on these samples
            mse = np.mean((images_np - recon_np) ** 2)
            logger.info(f"Epoch {epoch} - Batch MSE: {mse:.6f}")


if __name__ == "__main__":
    print("Phase1TrainWorker loaded successfully")
