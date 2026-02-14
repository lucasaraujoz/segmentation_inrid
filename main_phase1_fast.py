"""
SSMD-UNet Phase 1: FAST VERSION with pre-processed images
Main orchestration script
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import sys
from pathlib import Path
import argparse

# Add project paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.ssmd_unet_optimized import SSMDUNetPhase1
from data_factory.eyepacs_fast_dataset import EyePACSFastDataset
from train_and_val_worker_phase1 import Phase1TrainWorker
from configs.phase1_config import Phase1Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_phase1_fast(config: Phase1Config):
    """
    PHASE 1: FAST VERSION - Uses pre-processed images
    """
    
    logger.info(config)
    
    # ==================== PASSO 1: LOAD PREPROCESSED DATASET ====================
    logger.info("="*80)
    logger.info("PASSO 1: Carregando Dataset Pré-processado (RÁPIDO)")
    logger.info("="*80)
    
    data_dir = "/home/lucas/mestrado/tapi_inrid/eye_pacs/data/train_preprocessed"
    
    try:
        dataset = EyePACSFastDataset(data_dir)
    except ValueError as e:
        logger.error(str(e))
        logger.error("\nPré-processamento necessário! Execute:")
        logger.error("  python preprocess_eyepacs.py")
        logger.error("\nIsso vai levar ~30 min (única vez), depois treino será 10x mais rápido!")
        return
    
    logger.info(f"Dataset carregado: {len(dataset)} imagens")
    
    # Test sample
    sample = dataset[0]
    logger.info(f"Sample shape: {sample['image'].shape}")
    logger.info(f"Sample dtype: {sample['image'].dtype}")
    logger.info(f"Sample range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    
    # ==================== PASSO 2: CREATE DATALOADER ====================
    logger.info("\n" + "="*80)
    logger.info("PASSO 2: Criando DataLoader (OTIMIZADO)")
    logger.info("="*80)
    
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=True
    )
    
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Num workers: {config.num_workers}")
    logger.info(f"Total batches por epoch: {len(train_loader)}")
    logger.info(f"Expected time per epoch: ~{len(train_loader) * 0.45 / 60:.1f} min (optimized model)")
    
    # ==================== PASSO 3: CREATE MODEL ====================
    logger.info("\n" + "="*80)
    logger.info("PASSO 3: Criando SSMD-UNet Phase 1 (Autoencoder)")
    logger.info("="*80)
    
    model = SSMDUNetPhase1(
        in_channels=config.in_channels,
        out_channels=config.out_channels
    )
    
    logger.info("Arquitetura do Encoder:")
    logger.info(f"  Input: 512×512×3")
    logger.info(f"  Encoder blocks: [32, 64, 128, 256, 512] (OPTIMIZED - 2.4x faster)")
    logger.info(f"  Output (latent): 32×32×512")
    logger.info("Arquitetura do Decoder (Reconstruction):")
    logger.info(f"  Input (latent): 32×32×512")
    logger.info(f"  Decoder blocks: [256, 128, 64, 32]")
    logger.info(f"  Output: 512×512×3")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # ==================== PASSO 4: CREATE TRAINER ====================
    logger.info("\n" + "="*80)
    logger.info("PASSO 4: Criando Trainer")
    logger.info("="*80)
    
    trainer = Phase1TrainWorker(
        model=model,
        device=config.device,
        checkpoint_dir=config.checkpoint_dir
    )
    
    logger.info(f"Checkpoint dir: {config.checkpoint_dir}")
    
    # ==================== PASSO 5: TRAIN ====================
    logger.info("\n" + "="*80)
    logger.info("PASSO 5: Iniciando Treinamento")
    logger.info("="*80)
    logger.info(f"Loss: MSE (Mean Squared Error)")
    logger.info(f"Optimizer: SGD")
    logger.info(f"  - Learning rate: {config.learning_rate}")
    logger.info(f"  - Momentum: {config.momentum}")
    logger.info(f"Total epochs: {config.num_epochs}")
    logger.info(f"Checkpoint interval: {config.checkpoint_interval} epochs")
    logger.info(f"Visualization interval: {config.vis_interval} epochs")
    logger.info(f"Estimated time: ~{len(train_loader) * 0.45 * config.num_epochs / 3600:.1f} hours (~{len(train_loader) * 0.45 * config.num_epochs / 3600 / 24:.1f} days)")
    
    trainer.train(
        train_loader=train_loader,
        val_loader=None,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        checkpoint_interval=config.checkpoint_interval,
        vis_interval=config.vis_interval,
    )
    
    logger.info("\n" + "="*80)
    logger.info("✓ Training completed!")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSMD-UNet Phase 1 Training (FAST)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--workers", type=int, default=12, help="Number of workers")
    
    args = parser.parse_args()
    
    config = Phase1Config(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        num_workers=args.workers
    )
    
    run_phase1_fast(config)
