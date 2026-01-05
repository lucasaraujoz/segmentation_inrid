"""Test ASPP implementation before training."""

import torch
from models.unet_aspp import UNetWithASPP, ASPPModule
from configs.config import Config


def test_aspp_module():
    """Test ASPP module standalone."""
    print("=" * 80)
    print("TEST 1: ASPP Module")
    print("=" * 80)
    
    # Create ASPP module
    aspp = ASPPModule(
        in_channels=448,  # EfficientNet-B4 bottleneck
        out_channels=256,
        rates=[6, 12, 18]
    )
    
    # Test input (bottleneck feature map)
    x = torch.randn(2, 448, 32, 32)  # [B, C, H, W]
    
    print(f"Input shape:  {x.shape}")
    
    # Forward pass
    output = aspp(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected:     torch.Size([2, 256, 32, 32])")
    
    # Count parameters
    params = sum(p.numel() for p in aspp.parameters())
    print(f"\nASPP parameters: {params:,}")
    
    # Test gradient flow
    loss = output.mean()
    loss.backward()
    
    print(f"✅ Gradients computed successfully")
    
    return True


def test_unet_aspp():
    """Test full UNet with ASPP."""
    print("\n" + "=" * 80)
    print("TEST 2: UNet with ASPP")
    print("=" * 80)
    
    config = Config()
    config.aspp_rates = [6, 12, 18]
    config.aspp_channels = 256
    
    # Create model
    model = UNetWithASPP(
        encoder_name="efficientnet-b4",
        encoder_weights=None,  # No pretrained for testing
        in_channels=3,
        classes=2,
        aspp_rates=config.aspp_rates,
        aspp_channels=config.aspp_channels
    )
    
    # Get model stats
    params = model.get_num_parameters()
    
    print(f"Total parameters:      {params['total']:,}")
    print(f"Trainable parameters:  {params['trainable']:,}")
    print(f"ASPP parameters:       {params['aspp']:,} ({params['aspp_percentage']:.2f}%)")
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)  # [B, 3, H, W]
    
    print(f"\nInput shape:  {x.shape}")
    
    output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected:     torch.Size([2, 2, 512, 512])")
    
    # Test gradient flow
    loss = output.mean()
    loss.backward()
    
    print(f"✅ Gradients flow through entire model")
    
    return True


def test_with_real_config():
    """Test with actual training configuration."""
    print("\n" + "=" * 80)
    print("TEST 3: Real Training Configuration")
    print("=" * 80)
    
    from data_factory.data_factory import DataFactory
    from data_factory.ROP_dataset import ROPDataset
    
    config = Config()
    config.model_name = "unet_aspp"
    config.aspp_rates = [6, 12, 18]
    config.aspp_channels = 256
    
    # Load data
    data_factory = DataFactory(config)
    train_df, _ = data_factory.create_metadata_dataframe()
    
    # Get one sample
    sample_df = train_df.head(1)
    dataset = ROPDataset(
        dataframe=sample_df,
        config=config,
        is_train=False
    )
    
    sample = dataset[0]
    image = sample['image'].unsqueeze(0)  # [1, 3, H, W]
    mask = sample['mask'].unsqueeze(0)    # [1, 2, H, W]
    
    print(f"Image shape: {image.shape}")
    print(f"Mask shape:  {mask.shape}")
    
    # Create model
    from models.unet_aspp import create_unet_aspp
    model = create_unet_aspp(config)
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        output = model(image)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test with loss
    import segmentation_models_pytorch as smp
    dice_loss = smp.losses.DiceLoss(mode='multilabel', from_logits=True)
    
    output_train = model(image)
    loss = dice_loss(output_train, mask)
    
    print(f"\nDice loss: {loss.item():.4f}")
    print(f"✅ Model works with real data and loss")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "🧪" * 40)
    print("ASPP IMPLEMENTATION TESTS")
    print("🧪" * 40 + "\n")
    
    all_passed = True
    
    try:
        all_passed &= test_aspp_module()
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_unet_aspp()
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_with_real_config()
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nASPP is ready for training.")
        print("Run: python train_aspp.py")
    else:
        print("❌ SOME TESTS FAILED!")
        print("=" * 80)
        print("\nFix errors before training.")
    print()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
