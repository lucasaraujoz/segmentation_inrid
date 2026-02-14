"""
SSMD-UNet Phase 1: Unsupervised Pre-training with EyePACS
Main orchestration script following AGENT.md principles
Refactored to use configuration
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.ssmd_unet import SSMDUNetPhase1
from data_factory.eyepacs_dataset import EyePACSDataFactory, EyePACSDataset
from train_and_val_worker_phase1 import Phase1TrainWorker
from configs.phase1_config import Phase1Config, DEFAULT_CONFIG, PRODUCTION_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_phase1(config: Phase1Config):
    """
    PHASE 1: Unsupervised Pre-training (Autoencoder)
    
    Objetivo: Treinar encoder para aprender features gerais de retina SEM máscaras
    Dataset: EyePACS (35,126 imagens)
    Tempo: ~24-30h (conforme artigo com Batch 16, 100 epochs)
    """
    
    # Print configuration
    logger.info(config)
    
    # ==================== PASSO 1: LOAD DATASET ====================
    logger.info("="*80)
    logger.info("PASSO 1: Carregando Dataset EyePACS")
    logger.info("="*80)
    
    factory = EyePACSDataFactory(config.data_root)
    df = factory.scan_dataset()
    
    logger.info(f"Total de imagens encontradas: {len(df)}")
    logger.info(f"Primeiras 5 imagens:")
    for idx, row in df.head().iterrows():
        logger.info(f"  {row['image_name']}")
    
    # ==================== PASSO 2: CREATE DATASET ====================
    logger.info("\n" + "="*80)
    logger.info("PASSO 2: Criando Dataset com Pré-processamento")
    logger.info("="*80)
    logger.info("Pré-processamento pipeline:")
    logger.info(f"  1. Crop: Remove bordas pretas")
    logger.info(f"  2. Blur: Gaussian Blur (kernel={config.blur_kernel})")
    logger.info(f"  3. Resize: {config.image_size}×{config.image_size}")
    logger.info(f"  4. Normalize: [0, 1]")
    
    dataset = EyePACSDataset(
        df,
        image_size=config.image_size,
        apply_blur=config.apply_blur,
        blur_kernel=config.blur_kernel,
        normalize=config.normalize,
    )
    
    logger.info(f"Dataset criado: {len(dataset)} imagens")
    
    # Test sample
    sample = dataset[0]
    logger.info(f"Sample shape: {sample['image'].shape}")
    logger.info(f"Sample dtype: {sample['image'].dtype}")
    logger.info(f"Sample range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    
    # ==================== PASSO 3: CREATE DATALOADER ====================
    logger.info("\n" + "="*80)
    logger.info("PASSO 3: Criando DataLoader")
    logger.info("="*80)
    
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory if config.device == "cuda" else False,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Num workers: {config.num_workers}")
    logger.info(f"Total batches por epoch: {len(train_loader)}")
    logger.info(f"Total imagens por epoch: {len(train_loader) * config.batch_size}")
    logger.info(f"Expected time per epoch: ~{len(train_loader) * 0.2:.0f}s (~{len(train_loader) * 0.2 / 60:.1f} min)")
    
    # ==================== PASSO 4: CREATE MODEL ====================
    logger.info("\n" + "="*80)
    logger.info("PASSO 4: Criando SSMD-UNet Phase 1 (Autoencoder)")
    logger.info("="*80)
    
    model = SSMDUNetPhase1(
        in_channels=config.in_channels,
        out_channels=config.out_channels
    )
    
    # Print architecture summary
    logger.info("Arquitetura do Encoder:")
    logger.info(f"  Input: {config.image_size}×{config.image_size}×{config.in_channels}")
    logger.info(f"  Encoder blocks: {config.encoder_channels}")
    logger.info(f"  Output (latent): 16×16×1024")
    
    logger.info("Arquitetura do Decoder (Reconstruction):")
    logger.info(f"  Input (latent): 16×16×1024")
    logger.info(f"  Decoder blocks: [512, 256, 128, 64, 32]")
    logger.info(f"  Output: {config.image_size}×{config.image_size}×{config.out_channels}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # ==================== PASSO 5: CREATE TRAINER ====================
    logger.info("\n" + "="*80)
    logger.info("PASSO 5: Criando Trainer")
    logger.info("="*80)
    
    trainer = Phase1TrainWorker(
        model=model,
        device=config.device,
        checkpoint_dir=config.checkpoint_dir,
    )
    
    logger.info(f"Checkpoint dir: {config.checkpoint_dir}")
    
    # ==================== PASSO 6: TRAIN ====================
    logger.info("\n" + "="*80)
    logger.info("PASSO 6: Iniciando Treinamento")
    logger.info("="*80)
    logger.info(f"Loss: MSE (Mean Squared Error)")
    logger.info(f"Optimizer: SGD")
    logger.info(f"  - Learning rate: {config.learning_rate}")
    logger.info(f"  - Momentum: {config.momentum}")
    logger.info(f"Total epochs: {config.num_epochs}")
    logger.info(f"Checkpoint interval: {config.checkpoint_interval} epochs")
    logger.info(f"Visualization interval: {config.vis_interval} epochs")
    logger.info(f"Estimated time: ~24-30 hours (conforme artigo)")
    
    trainer.train(
        train_loader=train_loader,
        val_loader=None,  # No validation split for Phase 1
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        checkpoint_interval=config.checkpoint_interval,
        vis_interval=config.vis_interval,
    )
    
    logger.info("\n" + "="*80)
    logger.info("TREINO COMPLETADO!")
    logger.info("="*80)
    logger.info(f"Encoder final: {config.checkpoint_dir}/encoder_final.pth")
    logger.info(f"Decoder final: {config.checkpoint_dir}/decoder_rec_final.pth")
    logger.info(f"Checkpoint final: {config.checkpoint_dir}/phase1_final.pth")
    logger.info(f"Visualizações: {config.checkpoint_dir}/visualizations/")
    logger.info("\n✓ Pronto para Fase 2!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SSMD-UNet Phase 1 Training")
    parser.add_argument(
        "--config",
        choices=["default", "quick", "small", "production"],
        default="default",
        help="Configuration preset to use"
    )
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    
    args = parser.parse_args()
    
    # Select configuration preset
    if args.config == "default":
        config = DEFAULT_CONFIG
    elif args.config == "quick":
        from configs.phase1_config import QUICK_TEST_CONFIG
        config = QUICK_TEST_CONFIG
    elif args.config == "small":
        from configs.phase1_config import SMALL_CONFIG
        config = SMALL_CONFIG
    elif args.config == "production":
        config = PRODUCTION_CONFIG
    
    # Override with command line arguments
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    
    # Run training
    run_phase1(config)
