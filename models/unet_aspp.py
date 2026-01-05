"""Custom UNet with ASPP in the bottleneck.

ASPP (Atrous Spatial Pyramid Pooling) provides multi-scale context
by using dilated convolutions with different rates.

This is particularly useful for:
- Capturing lesions at different scales (small hemorrhages vs larger exudates)
- Preserving spatial resolution while increasing receptive field
- Adding minimal parameters compared to deeper networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import List, Optional


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling module.
    
    Uses parallel dilated convolutions with different rates to capture
    multi-scale context.
    
    Based on DeepLabV3+:
    "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
    https://arxiv.org/abs/1802.02611
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        rates: List[int] = [6, 12, 18]
    ):
        """Initialize ASPP module.
        
        Args:
            in_channels: Number of input channels (from encoder bottleneck)
            out_channels: Number of output channels
            rates: Dilation rates for atrous convolutions
        """
        super(ASPPModule, self).__init__()
        
        self.rates = rates
        
        # 1x1 convolution (captures global context)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Atrous convolutions with different rates
        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels, 3,
                        padding=rate, dilation=rate, bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion convolution
        # Input: 1x1 + len(rates) atrous + 1 global = (2 + len(rates)) branches
        num_branches = 2 + len(rates)
        self.project = nn.Sequential(
            nn.Conv2d(num_branches * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)  # Regularization
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, out_channels, H, W]
        """
        _, _, h, w = x.shape
        
        # 1x1 convolution
        feat1x1 = self.conv1x1(x)
        
        # Atrous convolutions
        atrous_feats = []
        for atrous_conv in self.atrous_convs:
            atrous_feats.append(atrous_conv(x))
        
        # Global average pooling
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(
            global_feat, size=(h, w),
            mode='bilinear', align_corners=False
        )
        
        # Concatenate all features
        concat_feats = torch.cat(
            [feat1x1] + atrous_feats + [global_feat],
            dim=1
        )
        
        # Project to output channels
        output = self.project(concat_feats)
        
        return output


class UNetWithASPP(nn.Module):
    """UNet with ASPP in the bottleneck.
    
    This replaces the standard bottleneck with an ASPP module to capture
    multi-scale context before the decoder.
    """
    
    def __init__(
        self,
        encoder_name: str = "efficientnet-b4",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 2,
        aspp_rates: List[int] = [6, 12, 18],
        aspp_channels: int = 256
    ):
        """Initialize UNet with ASPP.
        
        Args:
            encoder_name: Name of encoder (e.g., 'efficientnet-b4')
            encoder_weights: Pretrained weights ('imagenet' or None)
            in_channels: Number of input channels (3 for RGB)
            classes: Number of output classes
            aspp_rates: Dilation rates for ASPP
            aspp_channels: Number of channels in ASPP output
        """
        super(UNetWithASPP, self).__init__()
        
        # Create base UNet model
        self.base_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None  # We'll use sigmoid in loss
        )
        
        # Get encoder output channels (bottleneck)
        # For EfficientNet-B4: encoder.out_channels = [3, 24, 32, 56, 160, 448]
        # Bottleneck is the last one: 448
        encoder_channels = self.base_model.encoder.out_channels
        bottleneck_channels = encoder_channels[-1]
        
        print(f"Encoder channels: {encoder_channels}")
        print(f"Bottleneck channels: {bottleneck_channels}")
        
        # Create ASPP module
        self.aspp = ASPPModule(
            in_channels=bottleneck_channels,
            out_channels=aspp_channels,
            rates=aspp_rates
        )
        
        # Adjust decoder input to match ASPP output
        # We need to modify the decoder's first layer
        # Store original decoder
        self.decoder = self.base_model.decoder
        
        # Replace decoder's first center block to accept ASPP output
        # The decoder expects bottleneck_channels, but ASPP outputs aspp_channels
        # We'll add a projection layer
        self.aspp_projection = nn.Conv2d(
            aspp_channels, bottleneck_channels, 1
        )
        
        # Use the rest of base model
        self.segmentation_head = self.base_model.segmentation_head
        self.encoder = self.base_model.encoder
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            Segmentation output [B, classes, H, W]
        """
        # Encoder
        features = self.encoder(x)
        
        # Apply ASPP to bottleneck (last feature map)
        bottleneck = features[-1]
        aspp_output = self.aspp(bottleneck)
        
        # Project back to expected channels
        aspp_projected = self.aspp_projection(aspp_output)
        
        # Replace bottleneck with ASPP output
        features = list(features[:-1]) + [aspp_projected]
        
        # Decoder expects a list of features
        decoder_output = self.decoder(features)
        
        # Segmentation head
        masks = self.segmentation_head(decoder_output)
        
        return masks
    
    def get_num_parameters(self) -> dict:
        """Get number of parameters in the model.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        aspp_params = sum(p.numel() for p in self.aspp.parameters())
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "aspp": aspp_params,
            "aspp_percentage": (aspp_params / total_params) * 100
        }


def create_unet_aspp(config) -> UNetWithASPP:
    """Factory function to create UNet with ASPP.
    
    Args:
        config: Configuration object
        
    Returns:
        UNet with ASPP model
    """
    model = UNetWithASPP(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=3,
        classes=config.num_classes,
        aspp_rates=config.aspp_rates,
        aspp_channels=config.aspp_channels
    )
    
    # Print model statistics
    params = model.get_num_parameters()
    print("\n=== Model Statistics ===")
    print(f"Total parameters:      {params['total']:,}")
    print(f"Trainable parameters:  {params['trainable']:,}")
    print(f"ASPP parameters:       {params['aspp']:,} ({params['aspp_percentage']:.2f}%)")
    print("========================\n")
    
    return model
