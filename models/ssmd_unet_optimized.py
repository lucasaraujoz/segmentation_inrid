"""
SSMD-UNet: OPTIMIZED VERSION
Simple and fast U-Net without redundant interpolations
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Double convolution block with batch normalization and ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Simple upsampling (no ConvTranspose)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SSMDUNetEncoder(nn.Module):
    """
    Optimized U-Net Encoder with SMALLER channels for faster training
    Input: 512×512×3
    Output: 32×32×512 + skip connections
    Channels: [32, 64, 128, 256, 512] (half of original for 2.4x speedup)
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

    def forward(self, x):
        x1 = self.inc(x)      # 512×512×32
        x2 = self.down1(x1)   # 256×256×64
        x3 = self.down2(x2)   # 128×128×128
        x4 = self.down3(x3)   # 64×64×256
        x5 = self.down4(x4)   # 32×32×512
        
        return x5, [x4, x3, x2, x1]


class SSMDUNetDecoderRec(nn.Module):
    """
    Optimized U-Net Decoder for Reconstruction
    Input: 32×32×512
    Output: 512×512×3
    """
    def __init__(self, out_channels=3):
        super().__init__()
        self.up1 = Up(512 + 256, 256)
        self.up2 = Up(256 + 128, 128)
        self.up3 = Up(128 + 64, 64)
        self.up4 = Up(64 + 32, 32)
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x, skips):
        x = self.up1(x, skips[0])   # 64×64×256
        x = self.up2(x, skips[1])   # 128×128×128
        x = self.up3(x, skips[2])   # 256×256×64
        x = self.up4(x, skips[3])   # 512×512×32
        x = self.outc(x)            # 512×512×3
        x = torch.sigmoid(x)        # [0, 1]
        return x


class SSMDUNetDecoderSeg(nn.Module):
    """
    Optimized U-Net Decoder for Segmentation
    Input: 32×32×512
    Output: 512×512×1
    """
    def __init__(self):
        super().__init__()
        self.up1 = Up(512 + 256, 256)
        self.up2 = Up(256 + 128, 128)
        self.up3 = Up(128 + 64, 64)
        self.up4 = Up(64 + 32, 32)
        self.outc = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, skips):
        x = self.up1(x, skips[0])
        x = self.up2(x, skips[1])
        x = self.up3(x, skips[2])
        x = self.up4(x, skips[3])
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x


class SSMDUNetPhase1(nn.Module):
    """
    SSMD-UNet Phase 1: OPTIMIZED Autoencoder
    - 2.4x faster than original (33 min vs 78 min per epoch)
    - 4x smaller (7.8M vs 31M parameters)
    - Still effective for pretraining
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.encoder = SSMDUNetEncoder(in_channels)
        self.decoder_rec = SSMDUNetDecoderRec(out_channels)
    
    def forward(self, x):
        latent, skips = self.encoder(x)
        reconstruction = self.decoder_rec(latent, skips)
        return reconstruction
    
    def get_encoder(self):
        return self.encoder
    
    def get_decoder_rec(self):
        return self.decoder_rec


class SSMDUNetPhase2(nn.Module):
    """
    SSMD-UNet Phase 2: Multi-task Learning
    """
    def __init__(self, encoder, decoder_rec, num_lesions=4):
        super().__init__()
        self.encoder = encoder
        self.decoder_rec = decoder_rec
        
        # 4 segmentation decoders
        self.segmentation_decoders = nn.ModuleList([
            SSMDUNetDecoderSeg() for _ in range(num_lesions)
        ])
        
        self.lesion_names = ['HE', 'MA', 'EX', 'SE']
    
    def forward(self, x):
        latent, skips = self.encoder(x)
        reconstruction = self.decoder_rec(latent, skips)
        
        segmentations = []
        for decoder in self.segmentation_decoders:
            seg = decoder(latent, skips)
            segmentations.append(seg)
        
        return reconstruction, segmentations


if __name__ == "__main__":
    # Test model
    model = SSMDUNetPhase1(3, 3).cuda()
    x = torch.randn(2, 3, 512, 512).cuda()
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
