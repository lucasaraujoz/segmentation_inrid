"""
SSMD-UNet: FULL VERSION for 24GB GPU
Uses original large channels [64, 128, 256, 512, 1024]
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
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SSMDUNetEncoder(nn.Module):
    """
    FULL U-Net Encoder for 24GB GPU
    Channels: [64, 128, 256, 512, 1024]
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

    def forward(self, x):
        x1 = self.inc(x)      # 512×512×64
        x2 = self.down1(x1)   # 256×256×128
        x3 = self.down2(x2)   # 128×128×256
        x4 = self.down3(x3)   # 64×64×512
        x5 = self.down4(x4)   # 32×32×1024
        
        return x5, [x4, x3, x2, x1]


class SSMDUNetDecoderRec(nn.Module):
    """FULL Decoder for Reconstruction"""
    def __init__(self, out_channels=3):
        super().__init__()
        self.up1 = Up(1024 + 512, 512)
        self.up2 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, skips):
        x = self.up1(x, skips[0])   # 64×64×512
        x = self.up2(x, skips[1])   # 128×128×256
        x = self.up3(x, skips[2])   # 256×256×128
        x = self.up4(x, skips[3])   # 512×512×64
        x = self.outc(x)            # 512×512×3
        x = torch.sigmoid(x)
        return x


class SSMDUNetDecoderSeg(nn.Module):
    """FULL Decoder for Segmentation"""
    def __init__(self):
        super().__init__()
        self.up1 = Up(1024 + 512, 512)
        self.up2 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

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
    FULL Model for 24GB GPU
    31M parameters
    Better capacity for learning
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
    """Phase 2: Multi-task Learning"""
    def __init__(self, encoder, decoder_rec, num_lesions=4):
        super().__init__()
        self.encoder = encoder
        self.decoder_rec = decoder_rec
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
    model = SSMDUNetPhase1(3, 3).cuda()
    x = torch.randn(2, 3, 512, 512).cuda()
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
