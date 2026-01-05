#!/usr/bin/env python3
"""
Experiment 12: Wavelet-Enhanced Skip Connections

Minimal modification of baseline B4:
- Add wavelet decomposition (DWT) to encoder skip features
- Concatenate LL, LH, HL, HH to decoder
- Fixed wavelet (no learning), single level
- Target: improve hemorrhage reconstruction

Expected gain: +1-2% Dice (0.65-0.66 total)
"""

import sys
import json
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import pywt
import numpy as np

from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset import ROPDataset
from train_and_val_worker import TrainAndEvalWorker
from utils.utils import set_seed


class WaveletSkipConnection(nn.Module):
    """
    Apply DWT to skip connection features before concat with decoder.
    
    Fixed wavelet (db2), single level.
    Compress 4 subbands (LL, LH, HL, HH) back to original channels.
    """
    def __init__(self, in_channels, wavelet='db2'):
        super().__init__()
        self.wavelet = wavelet
        
        # 1x1 conv to compress 4*C channels back to C channels
        self.compress = nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, bias=False)
        
        # Initialize compression layer
        nn.init.xavier_uniform_(self.compress.weight)
    
    def dwt2d_batch(self, x):
        """Apply 2D DWT to batch of feature maps."""
        batch_size, channels, height, width = x.shape
        
        # Process each channel separately
        LL_list, LH_list, HL_list, HH_list = [], [], [], []
        
        for b in range(batch_size):
            for c in range(channels):
                # Extract single channel
                img = x[b, c].detach().cpu().numpy()
                
                # DWT
                coeffs = pywt.dwt2(img, self.wavelet)
                LL, (LH, HL, HH) = coeffs
                
                LL_list.append(torch.from_numpy(LL))
                LH_list.append(torch.from_numpy(LH))
                HL_list.append(torch.from_numpy(HL))
                HH_list.append(torch.from_numpy(HH))
        
        # Stack all channels and batches
        device = x.device
        h_new, w_new = LL_list[0].shape
        
        LL = torch.stack(LL_list).view(batch_size, channels, h_new, w_new).to(device)
        LH = torch.stack(LH_list).view(batch_size, channels, h_new, w_new).to(device)
        HL = torch.stack(HL_list).view(batch_size, channels, h_new, w_new).to(device)
        HH = torch.stack(HH_list).view(batch_size, channels, h_new, w_new).to(device)
        
        return LL, LH, HL, HH
    
    def forward(self, x):
        """
        Args:
            x: skip features (B, C, H, W)
        Returns:
            enhanced features (B, C, H/2, W/2) after wavelet
        """
        # DWT decomposition
        LL, LH, HL, HH = self.dwt2d_batch(x)
        
        # Concatenate all 4 subbands
        x_wavelet = torch.cat([LL, LH, HL, HH], dim=1)  # (B, 4*C, H/2, W/2)
        
        # Compress back to C channels
        x_out = self.compress(x_wavelet)  # (B, C, H/2, W/2)
        
        return x_out


class UNetWithWaveletSkips(nn.Module):
    """
    UNet with wavelet-enhanced skip connections.
    
    Based on segmentation_models_pytorch UNet + EfficientNet-B4.
    Inserts wavelet processing in skip paths.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        # Add wavelet skip processors
        # UNet typically has 4-5 skip connections
        # We'll add wavelet to the middle ones (most spatial detail)
        
        # Get encoder channels (EfficientNet-B4)
        encoder_channels = base_model.encoder.out_channels  # [3, 24, 32, 56, 160, 448]
        
        # Add wavelet to skips 2, 3, 4 (skip 1 and 5 are too extreme)
        self.wavelet_skip2 = WaveletSkipConnection(encoder_channels[2])
        self.wavelet_skip3 = WaveletSkipConnection(encoder_channels[3])
        self.wavelet_skip4 = WaveletSkipConnection(encoder_channels[4])
    
    def forward(self, x):
        """Forward with wavelet-enhanced skips."""
        # This is conceptual - actual implementation needs to hook into UNet decoder
        # For now, we'll use base model as-is and modify during training
        # Full implementation requires modifying segmentation_models_pytorch internals
        return self.base_model(x)


def main():
    print("="*80)
    print("EXPERIMENT 12: Wavelet-Enhanced Skip Connections")
    print("="*80)
    print()
    print("Baseline (B4): 0.6428 Test Dice")
    print("Modification: DWT (db2) on skip connections")
    print("Target: 0.65+ Test Dice (+2% improvement)")
    print()
    print("="*80)
    print()
    
    # Initialize config
    config = Config()
    config.model_name = "unet"
    config.encoder_name = "efficientnet-b4"
    config.encoder_weights = "imagenet"
    config.resolution = 512
    config.batch_size = 8
    config.num_epochs = 50
    config.learning_rate = 1e-4
    
    # Override checkpoint dir
    config.checkpoint_dir = os.path.join(config.output_dir, "checkpoints", "wavelet_skips")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    set_seed(config.random_state)
    
    # Load data
    data_factory = DataFactory(config)
    train_df, test_df = data_factory.create_metadata_dataframe()
    
    print(f"✓ Training images: {len(train_df)}")
    print(f"✓ Test images: {len(test_df)}")
    print()
    
    # Load frozen splits
    splits_path = Path("outputs/cv_splits.json")
    with open(splits_path) as f:
        cv_data = json.load(f)
    
    print(f"✓ Loaded frozen splits (hash: {cv_data['metadata']['train_data_hash'][:16]}...)")
    print()
    
    print("⚠️  IMPLEMENTATION NOTE:")
    print("Full wavelet skip integration requires modifying segmentation_models_pytorch")
    print("decoder blocks. This is a substantial refactor.")
    print()
    print("ALTERNATIVE APPROACH (simpler, same effect):")
    print("Use wavelet-transformed features as auxiliary input channel")
    print()
    
    # TODO: Implement full wavelet skip architecture
    # This requires:
    # 1. Subclass UNet decoder blocks
    # 2. Insert wavelet processing before skip concat
    # 3. Adjust channel dimensions
    
    print("="*80)
    print("RECOMMENDATION:")
    print("="*80)
    print()
    print("Before full implementation, test simpler alternatives:")
    print("1. Green-channel auxiliary input (easier, might give same gain)")
    print("2. Spatial attention in decoder (proven technique)")
    print("3. Focal loss reweighting for hemorrhages")
    print()
    print("If these fail, implement full wavelet skips.")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
