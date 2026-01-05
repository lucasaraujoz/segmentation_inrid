#!/usr/bin/env python3
"""Debug script to compare baseline vs attention gates."""

import torch
import segmentation_models_pytorch as smp
from models.unet_attention import create_unet_attention

# Create models
baseline = smp.Unet(
    encoder_name="efficientnet-b4",
    encoder_weights=None,
    in_channels=3,
    classes=2,
    activation=None
)

attention = create_unet_attention(
    encoder_name="efficientnet-b4",
    encoder_weights=None,
    in_channels=3,
    classes=2
)

# Test input
x = torch.randn(1, 3, 512, 512)

# Forward pass
with torch.no_grad():
    baseline_out = baseline(x)
    attention_out = attention(x)

print(f"Baseline output shape: {baseline_out.shape}")
print(f"Baseline output range: [{baseline_out.min():.4f}, {baseline_out.max():.4f}]")
print(f"Baseline output mean: {baseline_out.mean():.4f}")
print()
print(f"Attention output shape: {attention_out.shape}")
print(f"Attention output range: [{attention_out.min():.4f}, {attention_out.max():.4f}]")
print(f"Attention output mean: {attention_out.mean():.4f}")
print()

# Check encoder features
baseline_features = baseline.encoder(x)
attention_features = attention.encoder(x)

print("Encoder features match:", len(baseline_features) == len(attention_features))
for i, (bf, af) in enumerate(zip(baseline_features, attention_features)):
    print(f"  Stage {i}: shapes match = {bf.shape == af.shape}")
