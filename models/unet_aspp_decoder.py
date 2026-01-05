"""
UNet with ASPP (Atrous Spatial Pyramid Pooling) in Decoder blocks.

ASPP applies parallel dilated convolutions at multiple rates to capture
multi-scale context in the decoder upsampling path.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import List


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""
    
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: List[int] = [6, 12, 18]):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            atrous_rates: Dilation rates for parallel atrous convolutions
        """
        super().__init__()
        
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolutions with different rates
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Global average pooling branch
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        self.convs = nn.ModuleList(modules)
        
        # Projection layer to combine all branches
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        res = []
        for conv in self.convs:
            if isinstance(conv[0], nn.AdaptiveAvgPool2d):
                # Global pooling branch - upsample back to input size
                pooled = conv(x)
                pooled = nn.functional.interpolate(pooled, size=x.shape[2:], mode='bilinear', align_corners=False)
                res.append(pooled)
            else:
                res.append(conv(x))
        
        res = torch.cat(res, dim=1)
        return self.project(res)


class DecoderBlockWithASPP(nn.Module):
    """Decoder block with ASPP for multi-scale feature extraction."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_aspp: bool = True,
        atrous_rates: List[int] = [6, 12, 18]
    ):
        super().__init__()
        
        self.use_aspp = use_aspp
        
        # Standard convolution after upsampling + skip connection
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        if use_aspp:
            # ASPP module for multi-scale context
            self.aspp = ASPPModule(out_channels, out_channels, atrous_rates)
        
        # Final convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip=None):
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv1(x)
        
        if self.use_aspp:
            x = self.aspp(x)
        
        x = self.conv2(x)
        
        return x


class UNetASPPDecoder(nn.Module):
    """UNet with ASPP modules in decoder blocks."""
    
    def __init__(
        self,
        encoder_name: str = 'efficientnet-b4',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 2,
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        use_aspp_at_levels: List[int] = [0, 1, 2],  # Which decoder levels to use ASPP
        atrous_rates: List[int] = [6, 12, 18]
    ):
        """
        Args:
            encoder_name: Encoder architecture
            encoder_weights: Pretrained weights
            in_channels: Number of input channels
            classes: Number of output classes
            decoder_channels: Channels in each decoder block
            use_aspp_at_levels: Indices of decoder levels to apply ASPP (0=deepest)
            atrous_rates: Dilation rates for ASPP
        """
        super().__init__()
        
        # Use SMP encoder
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights
        )
        
        encoder_channels = self.encoder.out_channels
        
        # Build decoder with ASPP at specified levels
        self.decoder_blocks = nn.ModuleList()
        
        # Center (bottleneck)
        self.center = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], decoder_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            skip_ch = encoder_channels[-(i + 2)]
            out_ch = decoder_channels[i + 1]
            use_aspp = i in use_aspp_at_levels
            
            self.decoder_blocks.append(
                DecoderBlockWithASPP(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    use_aspp=use_aspp,
                    atrous_rates=atrous_rates
                )
            )
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        
        # Center
        x = self.center(features[-1])
        
        # Decoder with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = features[-(i + 2)]
            x = decoder_block(x, skip)
        
        # Final upsampling to input size
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Segmentation head
        x = self.segmentation_head(x)
        
        return x


def create_unet_aspp_decoder(config):
    """Factory function to create UNet with ASPP decoder."""
    return UNetASPPDecoder(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=3,
        classes=config.num_classes,
        decoder_channels=[256, 128, 64, 32, 16],
        use_aspp_at_levels=[0, 1, 2],  # ASPP at first 3 decoder levels
        atrous_rates=[6, 12, 18]
    )
