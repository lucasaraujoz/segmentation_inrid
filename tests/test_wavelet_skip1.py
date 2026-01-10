"""
Script de teste para validar modelo wavelet skip 1.
Testa arquitetura e pipeline completo antes do treinamento.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from models.unet_wavelet_skip1 import UnetWaveletSkip1
from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset import ROPDataset


def test_wavelet_module():
    """Teste 1: Módulo wavelet isolado"""
    print("=" * 80)
    print("TEST 1: Wavelet Skip Connection Module")
    print("=" * 80)
    
    from models.wavelet_skip import WaveletSkipConnection
    
    # Simular skip do EfficientNet-B4 (24 canais)
    skip_channels = 24
    batch_size = 2
    height, width = 256, 256
    
    wavelet_module = WaveletSkipConnection(in_channels=skip_channels)
    
    # Input tensor
    skip_input = torch.randn(batch_size, skip_channels, height, width)
    
    print(f"Input shape: {skip_input.shape}")
    print(f"Input channels: {skip_channels}")
    
    # Forward pass
    try:
        enhanced_skip = wavelet_module(skip_input)
        print(f"Output shape: {enhanced_skip.shape}")
        
        # Verificar shape
        assert enhanced_skip.shape == skip_input.shape, "Shape mismatch!"
        print("✓ Shape preserved correctly")
        
        # Verificar que saída não é igual à entrada (transformação aplicada)
        assert not torch.allclose(skip_input, enhanced_skip), "Output identical to input!"
        print("✓ Wavelet transformation applied")
        
        print("✓ TEST 1 PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ TEST 1 FAILED: {e}\n")
        return False


def test_model_architecture():
    """Teste 2: Arquitetura completa do modelo"""
    print("=" * 80)
    print("TEST 2: UNet Wavelet Skip 1 Architecture")
    print("=" * 80)
    
    try:
        # Criar modelo
        model = UnetWaveletSkip1(
            encoder_name='efficientnet-b4',
            encoder_weights=None,  # Sem pesos para teste rápido
            in_channels=3,
            classes=2
        )
        
        print(f"✓ Model instantiated")
        
        # Contar parâmetros
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 3, 512, 512)
        
        print(f"\nInput shape: {x.shape}")
        
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        print(f"Output shape: {output.shape}")
        
        # Verificar shape
        expected_shape = (batch_size, 2, 512, 512)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print("✓ Output shape correct")
        
        # Verificar range de valores
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        print("✓ TEST 2 PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ TEST 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_integration():
    """Teste 3: Integração com dataset"""
    print("=" * 80)
    print("TEST 3: Dataset Integration")
    print("=" * 80)
    
    try:
        # Config
        config = Config()
        config.img_size = 512
        config.apply_clahe = False
        config.batch_size = 4
        
        # Data
        data_factory = DataFactory(config)
        train_df, test_df = data_factory.create_metadata_dataframe()
        
        print(f"✓ Train images: {len(train_df)}")
        print(f"✓ Test images: {len(test_df)}")
        
        # Dataset
        dataset = ROPDataset(
            dataframe=train_df.head(4),
            config=config,
            is_train=False
        )
        
        print(f"✓ Dataset created: {len(dataset)} samples")
        
        # DataLoader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Modelo
        model = UnetWaveletSkip1(
            encoder_name='efficientnet-b4',
            encoder_weights=None,
            in_channels=3,
            classes=2
        )
        model.eval()
        
        print("\nTesting batch processing...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                images = batch['image']
                masks = batch['mask']
                
                print(f"Batch {batch_idx + 1}:")
                print(f"  Images: {images.shape}")
                print(f"  Masks: {masks.shape}")
                
                # Forward pass
                outputs = model(images)
                print(f"  Outputs: {outputs.shape}")
                
                assert outputs.shape == masks.shape, "Output/mask shape mismatch!"
                print("  ✓ Shapes match")
                
                if batch_idx >= 1:  # Testar 2 batches
                    break
        
        print("\n✓ TEST 3 PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ TEST 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage():
    """Teste 4: Uso de memória"""
    print("=" * 80)
    print("TEST 4: Memory Usage")
    print("=" * 80)
    
    try:
        import torch.cuda as cuda
        
        if not cuda.is_available():
            print("CUDA not available, skipping memory test")
            return True
        
        device = torch.device('cuda')
        
        # Limpar memória
        cuda.empty_cache()
        
        # Memória inicial
        mem_before = cuda.memory_allocated(device) / 1024**2
        print(f"Memory before model: {mem_before:.2f} MB")
        
        # Criar modelo
        model = UnetWaveletSkip1(
            encoder_name='efficientnet-b4',
            encoder_weights='imagenet',
            in_channels=3,
            classes=2
        ).to(device)
        
        mem_after_model = cuda.memory_allocated(device) / 1024**2
        print(f"Memory after model: {mem_after_model:.2f} MB")
        print(f"Model size: {mem_after_model - mem_before:.2f} MB")
        
        # Forward pass
        x = torch.randn(4, 3, 512, 512).to(device)
        
        model.eval()
        with torch.no_grad():
            y = model(x)
        
        mem_after_forward = cuda.memory_allocated(device) / 1024**2
        print(f"Memory after forward: {mem_after_forward:.2f} MB")
        print(f"Forward pass overhead: {mem_after_forward - mem_after_model:.2f} MB")
        
        # Cleanup
        del model, x, y
        cuda.empty_cache()
        
        print("✓ TEST 4 PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ TEST 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "WAVELET SKIP 1 - VALIDATION TESTS" + " " * 25 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")
    
    results = []
    
    # Run all tests
    results.append(("Wavelet Module", test_wavelet_module()))
    results.append(("Model Architecture", test_model_architecture()))
    results.append(("Dataset Integration", test_dataset_integration()))
    results.append(("Memory Usage", test_memory_usage()))
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Ready to train!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - Fix issues before training")
        return 1


if __name__ == "__main__":
    sys.exit(main())
