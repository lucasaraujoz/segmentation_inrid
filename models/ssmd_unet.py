"""
SSMD-UNet: Semi-Supervised Multi-task Decoders Network
Inspired by: Ullah et al. (2023) - Scientific Reports
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block with batch normalization and ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SSMDUNetEncoder(nn.Module):
    """
    U-Net Encoder (contracting path)
    Input: 512×512×3
    Output: 16×16×1024 (latent representation)
    Channels: [64, 128, 256, 512, 1024]
    """
    def __init__(self, in_channels=3, channels=[64, 128, 256, 512, 1024]):
        super().__init__()
        self.channels = channels
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        prev_channels = in_channels
        for ch in channels:
            self.downs.append(DoubleConv(prev_channels, ch))
            self.pools.append(nn.MaxPool2d(2))
            prev_channels = ch

    def forward(self, x):
        """
        Forward pass storing skip connections
        Returns:
            latent: 16×16×1024 feature map
            skips: list of 4 skip connections [512×512×64, 256×256×128, 128×128×256, 64×64×512]
        """
        skips = []
        
        # Encoder path: 512 -> 256 -> 128 -> 64 -> 32 -> 16
        for i, (down, pool) in enumerate(zip(self.downs, self.pools)):
            x = down(x)
            if i < len(self.downs) - 1:  # Don't add latent to skips
                skips.append(x)
            x = pool(x)
        
        # x is now latent (16×16×1024)
        return x, skips


class SSMDUNetDecoderRec(nn.Module):
    """
    U-Net Decoder for Reconstruction (expanding path)
    Input: 16×16×1024 (latent)
    Output: 512×512×3 (reconstructed image)
    """
    def __init__(self, out_channels=3):
        super().__init__()
        
        # Upsampling + Convolution layers
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up1 = DoubleConv(512 + 512, 512)
        
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = DoubleConv(256 + 256, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = DoubleConv(128 + 128, 128)
        
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4 = DoubleConv(64 + 64, 64)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, latent, skips):
        """
        Forward pass with skip connections
        skips: [512×512×64, 256×256×128, 128×128×256, 64×64×512] (in order from encoder)
        """
        x = latent  # 16×16×1024
        
        # Step 1: 16×16×1024 -> 32×32×512
        x = self.upconv1(x)  # -> 32×32×512
        skip = skips[3]  # 64×64×512
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.up1(x)  # -> 32×32×512
        
        # Step 2: 32×32×512 -> 64×64×256
        x = self.upconv2(x)  # -> 64×64×256
        skip = skips[2]  # 128×128×256
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.up2(x)  # -> 64×64×256
        
        # Step 3: 64×64×256 -> 128×128×128
        x = self.upconv3(x)  # -> 128×128×128
        skip = skips[1]  # 256×256×128
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.up3(x)  # -> 128×128×128
        
        # Step 4: 128×128×128 -> 256×256×64
        x = self.upconv4(x)  # -> 256×256×64
        skip = skips[0]  # 512×512×64
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.up4(x)  # -> 256×256×64
        
        # Final output: 256×256×64 -> 512×512×3
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        x = self.final_conv(x)  # -> 512×512×3
        
        # Sigmoid activation for image reconstruction [0, 1]
        x = torch.sigmoid(x)
        
        return x


class SSMDUNetDecoderSeg(nn.Module):
    """
    U-Net Decoder for Segmentation (expanding path)
    Input: 16×16×1024 (latent)
    Output: 512×512×1 (segmentation mask)
    """
    def __init__(self):
        super().__init__()
        
        # Upsampling + Convolution layers
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up1 = DoubleConv(512 + 512, 512)
        
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = DoubleConv(256 + 256, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = DoubleConv(128 + 128, 128)
        
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4 = DoubleConv(64 + 64, 64)
        
        # Final output layer (binary mask)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, latent, skips):
        """
        Forward pass with skip connections
        skips: [512×512×64, 256×256×128, 128×128×256, 64×64×512]
        """
        x = latent  # 16×16×1024
        
        # Step 1: 16×16×1024 -> 32×32×512
        x = self.upconv1(x)  # -> 32×32×512
        skip = skips[3]  # 64×64×512
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.up1(x)  # -> 32×32×512
        
        # Step 2: 32×32×512 -> 64×64×256
        x = self.upconv2(x)  # -> 64×64×256
        skip = skips[2]  # 128×128×256
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.up2(x)  # -> 64×64×256
        
        # Step 3: 64×64×256 -> 128×128×128
        x = self.upconv3(x)  # -> 128×128×128
        skip = skips[1]  # 256×256×128
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.up3(x)  # -> 128×128×128
        
        # Step 4: 128×128×128 -> 256×256×64
        x = self.upconv4(x)  # -> 256×256×64
        skip = skips[0]  # 512×512×64
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.up4(x)  # -> 256×256×64
        
        # Final output: 256×256×64 -> 512×512×1
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        x = self.final_conv(x)  # -> 512×512×1
        
        # Sigmoid activation for mask [0, 1]
        x = torch.sigmoid(x)
        
        return x


class SSMDUNetPhase1(nn.Module):
    """
    SSMD-UNet Phase 1: Unsupervised Autoencoder
    For pre-training encoder with reconstruction task
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.encoder = SSMDUNetEncoder(in_channels)
        self.decoder_rec = SSMDUNetDecoderRec(out_channels)
    
    def forward(self, x):
        latent, skips = self.encoder(x)
        reconstruction = self.decoder_rec(latent, skips)
        return reconstruction, latent, skips
    
    def get_encoder(self):
        return self.encoder
    
    def get_decoder_rec(self):
        return self.decoder_rec


class SSMDUNetPhase2(nn.Module):
    """
    SSMD-UNet Phase 2: Supervised Multi-task Learning
    Fine-tunes encoder + trains 4 segmentation decoders + reconstruction decoder
    """
    def __init__(self, encoder, decoder_rec, num_lesions=4):
        super().__init__()
        self.encoder = encoder
        self.decoder_rec = decoder_rec
        
        # 4 segmentation decoders (HE, MA, EX, SE)
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
        
        return reconstruction, segmentations, latent, skips
    
    def forward_primary(self, x, primary_idx):
        """Forward for specific primary task (0=HE, 1=MA, 2=EX, 3=SE)"""
        latent, skips = self.encoder(x)
        reconstruction = self.decoder_rec(latent, skips)
        
        segmentations = []
        for idx, decoder in enumerate(self.segmentation_decoders):
            seg = decoder(latent, skips)
            segmentations.append(seg)
        
        return reconstruction, segmentations, latent, skips


if __name__ == "__main__":
    # Test Phase 1
    print("Testing SSMD-UNet Phase 1...")
    model = SSMDUNetPhase1()
    x = torch.randn(2, 3, 512, 512)
    reconstruction, latent, skips = model(x)
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Reconstruction shape: {reconstruction.shape}")
    print(f"✓ Latent shape: {latent.shape}")
    print(f"✓ Number of skip connections: {len(skips)}")
    
    # Test Phase 2
    print("\nTesting SSMD-UNet Phase 2...")
    model_p2 = SSMDUNetPhase2(model.encoder, model.decoder_rec)
    reconstruction, segmentations, latent, skips = model_p2(x)
    print(f"✓ Reconstruction shape: {reconstruction.shape}")
    print(f"✓ Number of segmentations: {len(segmentations)}")
    print(f"✓ Models built successfully!")
