"""UNet with Attention Gates.

Attention Gates learn to focus on relevant regions (lesions) while
suppressing irrelevant background features in the skip connections.

Based on "Attention U-Net: Learning Where to Look for the Pancreas" (2018)
https://arxiv.org/abs/1804.03999

Key benefits:
- Focuses on lesions vs background (critical for sparse lesions like hemorrhages)
- Adds only ~10% parameters (vs 15% for ASPP)
- Proven effective on small medical datasets
- Works well with U-Net architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional


class AttentionGate(nn.Module):
    """Attention Gate module.
    
    Learns attention coefficients to highlight salient features
    and suppress irrelevant regions in skip connections.
    
    Uses gating signal from decoder to weight encoder features.
    """
    
    def __init__(
        self,
        gate_channels: int,  # Channels from decoder (gating signal)
        skip_channels: int,  # Channels from encoder (to be gated)
        inter_channels: Optional[int] = None
    ):
        """Initialize Attention Gate.
        
        Args:
            gate_channels: Number of channels in gating signal (from decoder)
            skip_channels: Number of channels in skip connection (from encoder)
            inter_channels: Intermediate channels (default: skip_channels // 2)
        """
        super(AttentionGate, self).__init__()
        
        if inter_channels is None:
            inter_channels = skip_channels // 2
            if inter_channels == 0:
                inter_channels = 1
        
        # Transform gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        # Transform skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        # Attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Skip connection features from encoder [B, skip_channels, H, W]
            g: Gating signal from decoder [B, gate_channels, H', W']
            
        Returns:
            Attention-weighted features [B, skip_channels, H, W]
        """
        # Get input sizes
        input_size = x.size()[2:]
        
        # Upsample gating signal to match skip connection size if needed
        if g.size()[2:] != input_size:
            g = F.interpolate(g, size=input_size, mode='bilinear', align_corners=False)
        
        # Transform inputs
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Combine and compute attention coefficients
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention to skip connection
        return x * psi


class UNetWithAttention(nn.Module):
    """UNet with Attention Gates in skip connections.
    
    Adds attention gates between encoder and decoder to focus on
    relevant regions (lesions) while suppressing background.
    
    Simple approach: Apply attention gates as a wrapper around base UNet.
    """
    
    def __init__(
        self,
        encoder_name: str = "efficientnet-b4",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 2
    ):
        """Initialize UNet with Attention Gates.
        
        Args:
            encoder_name: Name of encoder (e.g., 'efficientnet-b4')
            encoder_weights: Pretrained weights ('imagenet' or None)
            in_channels: Number of input channels (3 for RGB)
            classes: Number of output classes
        """
        super(UNetWithAttention, self).__init__()
        
        # Create base UNet model (we'll use it as reference for decoder structure)
        base_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None
        )
        
        # Use encoder from base model
        self.encoder = base_model.encoder
        
        # Get encoder channels
        # For EfficientNet-B4: [3, 48, 32, 56, 160, 448]
        # features[0] = input (not used as skip)
        # features[1-4] = encoder stages (used as skip connections)
        # features[5] = bottleneck
        encoder_channels = self.encoder.out_channels
        
        print(f"Encoder channels: {encoder_channels}")
        
        # Create custom decoder with attention gates
        # Decoder has 5 stages, but only 4 have skip connections (stages 1-4, not input)
        decoder_channels = (256, 128, 64, 32, 16)
        
        # Build decoder blocks with attention
        self.decoder_blocks = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        
        # Start from bottleneck
        in_ch = encoder_channels[-1]  # 448
        
        # Create 5 decoder blocks
        # Blocks 0-3 have skip connections from encoder stages 4,3,2,1
        # Block 4 has NO skip connection (no input as skip!)
        for i, out_ch in enumerate(decoder_channels):
            # Determine if this block has a skip connection
            # Only blocks 0-3 get skip from stages 4,3,2,1
            # Block 4 should NOT use input (features[0]) as skip
            if i < 4:  # First 4 blocks have skip connections
                skip_idx = 4 - i  # 4, 3, 2, 1 (NOT 0!)
                skip_ch = encoder_channels[skip_idx]
                
                # Add attention gate for this skip connection
                self.attention_gates.append(
                    AttentionGate(
                        gate_channels=in_ch,  # From previous decoder stage (gating signal)
                        skip_channels=skip_ch  # From encoder (to be gated)
                    )
                )
            else:
                # Last block: no skip connection
                skip_ch = 0
            
            # Decoder block: upsample + skip + conv
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
            
            in_ch = out_ch
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], classes, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention gates.
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            Segmentation output [B, classes, H, W]
        """
        # Encoder: extract multi-scale features
        features = self.encoder(x)
        # features[0]: input [B, 3, 512, 512] - NOT USED AS SKIP
        # features[1]: stage1 [B, 48, 256, 256]
        # features[2]: stage2 [B, 32, 128, 128]  
        # features[3]: stage3 [B, 56, 64, 64]
        # features[4]: stage4 [B, 160, 32, 32]
        # features[5]: bottleneck [B, 448, 16, 16]
        
        # Decoder: upsample with attention-gated skip connections
        x = features[-1]  # Start from bottleneck [B, 448, 16, 16]
        
        att_idx = 0
        for i in range(len(self.decoder_blocks)):
            # Blocks 0-3 have skip connections from stages 4,3,2,1
            # Block 4 has NO skip connection
            if i < 4:
                skip_idx = 4 - i  # 4, 3, 2, 1
                skip = features[skip_idx]
                
                # Apply attention gate: use current decoder feature as gate
                skip = self.attention_gates[att_idx](x=skip, g=x)
                att_idx += 1
                
                # Upsample current feature to match skip spatial size
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='nearest')
                
                # Concatenate upsampled feature with gated skip
                x = torch.cat([x, skip], dim=1)
            else:
                # Last block: no skip connection, just upsample
                x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
            
            # Apply decoder block (skip Upsample layer since we already upsampled)
            for layer in self.decoder_blocks[i]:
                if not isinstance(layer, nn.Upsample):
                    x = layer(x)
        
        # Final segmentation head
        masks = self.segmentation_head(x)
        
        return masks
    
    def get_num_parameters(self) -> dict:
        """Get number of parameters in the model.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Attention gates parameters
        attention_params = sum(
            sum(p.numel() for p in gate.parameters()) 
            for gate in self.attention_gates
        )
        
        # Base model parameters (encoder + decoder without attention)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder_blocks.parameters())
        head_params = sum(p.numel() for p in self.segmentation_head.parameters())
        base_params = encoder_params + decoder_params + head_params - attention_params
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "attention": attention_params,
            "attention_percentage": (attention_params / total_params) * 100,
            "base": base_params
        }


def create_unet_attention(
    encoder_name: str = "efficientnet-b4",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 2
):
    """Factory function to create UNet with Attention Gates.
    
    Args:
        encoder_name: Name of encoder (e.g., 'efficientnet-b4')
        encoder_weights: Pretrained weights ('imagenet' or None)
        in_channels: Number of input channels (3 for RGB)
        classes: Number of output classes
        
    Returns:
        UNet with Attention Gates model
    """
    model = UNetWithAttention(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )
    
    # Print model statistics
    params = model.get_num_parameters()
    print("\n=== Model Statistics ===")
    print(f"Total parameters:       {params['total']:,}")
    print(f"Trainable parameters:   {params['trainable']:,}")
    print(f"Attention parameters:   {params['attention']:,} ({params['attention_percentage']:.2f}%)")
    print("========================\n")
    
    return model
