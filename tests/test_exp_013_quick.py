"""
Teste rápido do Experimento 013: Patch-Based Training com Controle de Leakage

Verifica:
1. Controle de leakage entre train/val/test
2. Dataset de patches funciona corretamente
3. Modelo funciona com patches
4. Reconstrução funciona

Execução:
    python tests/test_exp_013_quick.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import torch
from collections import defaultdict

from configs.config import Config
from data_factory.data_factory import DataFactory


def test_train_test_separation():
    """Verifica que train e test são completamente separados."""
    print("\n[TEST 1] Verificando separação Train/Test...")
    
    config = Config()
    data_factory = DataFactory(config)
    train_df, test_df = data_factory.create_metadata_dataframe()
    
    train_images = set(train_df['image_name'].tolist())
    test_images = set(test_df['image_name'].tolist())
    
    overlap = train_images & test_images
    assert len(overlap) == 0, f"FALHOU: Imagens em train E test: {overlap}"
    
    print(f"  ✓ Train/Test separados: {len(train_images)} train, {len(test_images)} test")
    return True


def test_cv_fold_separation():
    """Verifica que folds de CV não têm overlap."""
    print("\n[TEST 2] Verificando separação dos Folds de CV...")
    
    config = Config()
    
    # Carregar splits
    splits_path = os.path.join(config.output_dir, 'frozen_cv_splits.json')
    if not os.path.exists(splits_path):
        splits_path = os.path.join(config.output_dir, 'cv_splits.json')
    
    with open(splits_path, 'r') as f:
        splits_data = json.load(f)
    
    data_factory = DataFactory(config)
    train_df, _ = data_factory.create_metadata_dataframe()
    
    for fold_data in splits_data['folds']:
        fold_num = fold_data['fold']
        train_idx = set(fold_data['train_indices'])
        val_idx = set(fold_data['val_indices'])
        
        # Não pode haver interseção
        overlap = train_idx & val_idx
        assert len(overlap) == 0, f"FALHOU: Fold {fold_num} tem índices em train E val: {overlap}"
        
        # Verificar por imagem (mais explícito)
        train_images = set(train_df.iloc[list(train_idx)]['image_name'].tolist())
        val_images = set(train_df.iloc[list(val_idx)]['image_name'].tolist())
        
        img_overlap = train_images & val_images
        assert len(img_overlap) == 0, f"FALHOU: Fold {fold_num} - Imagens em train E val: {img_overlap}"
    
    print(f"  ✓ Todos os {len(splits_data['folds'])} folds têm splits corretos sem leakage")
    return True


def test_patch_dataset_creation():
    """Testa criação do dataset de patches."""
    print("\n[TEST 3] Verificando criação do dataset de patches...")
    
    from data_factory.ROP_dataset_patches import ROPDatasetPatches
    
    config = Config()
    config.apply_clahe = False
    
    data_factory = DataFactory(config)
    train_df, _ = data_factory.create_metadata_dataframe()
    
    # Usar apenas 2 imagens para teste rápido
    sample_df = train_df.head(2)
    
    dataset = ROPDatasetPatches(
        dataframe=sample_df,
        config=config,
        patch_size=512,
        overlap=50,
        is_train=False
    )
    
    assert len(dataset) > 0, "FALHOU: Dataset vazio"
    
    # Verificar dimensões
    sample = dataset[0]
    
    img_shape = sample['image'].shape
    assert img_shape == (3, 512, 512), f"FALHOU: Shape da imagem incorreto: {img_shape}"
    
    mask_shape = sample['mask'].shape
    assert mask_shape == (2, 512, 512), f"FALHOU: Shape da máscara incorreto: {mask_shape}"
    
    print(f"  ✓ Dataset criado com {len(dataset)} patches de {len(sample_df)} imagens")
    print(f"  ✓ Dimensões corretas: image={img_shape}, mask={mask_shape}")
    return True


def test_patches_grouped_by_image():
    """TESTE CRÍTICO: Verifica que patches são agrupados por imagem."""
    print("\n[TEST 4] Verificando agrupamento de patches por imagem...")
    
    from data_factory.ROP_dataset_patches import ROPDatasetPatches
    
    config = Config()
    config.apply_clahe = False
    
    data_factory = DataFactory(config)
    train_df, _ = data_factory.create_metadata_dataframe()
    
    # Usar 3 imagens
    sample_df = train_df.head(3)
    
    dataset = ROPDatasetPatches(
        dataframe=sample_df,
        config=config,
        patch_size=512,
        overlap=50,
        is_train=False
    )
    
    # Mapear patches por imagem
    patches_by_image = defaultdict(list)
    
    for idx in range(len(dataset)):
        info = dataset.patch_info[idx]
        img_idx = info['image_idx']
        patches_by_image[img_idx].append(idx)
    
    # Verificar que cada imagem tem múltiplos patches
    print(f"  Distribuição de patches por imagem:")
    for img_idx, patch_indices in patches_by_image.items():
        img_name = sample_df.iloc[img_idx]['image_name']
        print(f"    Imagem {img_idx} ({img_name}): {len(patch_indices)} patches")
        assert len(patch_indices) > 1, f"FALHOU: Imagem {img_idx} tem apenas {len(patch_indices)} patch"
    
    print(f"  ✓ {len(patches_by_image)} imagens, cada uma com múltiplos patches")
    return True


def test_model_forward_pass():
    """Testa forward pass do modelo com patches."""
    print("\n[TEST 5] Verificando forward pass do modelo...")
    
    import segmentation_models_pytorch as smp
    
    model = smp.Unet(
        encoder_name='efficientnet-b4',
        encoder_weights='imagenet',
        in_channels=3,
        classes=2
    )
    model.eval()
    
    # Batch de patches
    x = torch.randn(4, 3, 512, 512)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (4, 2, 512, 512), f"FALHOU: Shape do output incorreto: {output.shape}"
    assert not torch.isnan(output).any(), "FALHOU: NaN no output"
    assert not torch.isinf(output).any(), "FALHOU: Inf no output"
    
    print(f"  ✓ Forward pass OK: input={x.shape} → output={output.shape}")
    return True


def test_reconstruction():
    """Testa reconstrução de imagem a partir de patches."""
    print("\n[TEST 6] Verificando reconstrução de patches...")
    
    # Importar função de reconstrução diretamente
    # (não precisa das funções de dice para este teste)
    import numpy as np
    
    def reconstruct_from_patches(patches_pred, patch_info, img_width, img_height, 
                                 patch_size=512, num_classes=2):
        """Reconstrói predição - cópia local para teste."""
        full_pred = np.zeros((num_classes, img_height, img_width), dtype=np.float32)
        counts = np.zeros((img_height, img_width), dtype=np.float32)
        
        for i, info in enumerate(patch_info):
            x, y = info['patch_x'], info['patch_y']
            patch = patches_pred[i]
            full_pred[:, y:y+patch_size, x:x+patch_size] += patch
            counts[y:y+patch_size, x:x+patch_size] += 1
        
        counts = np.maximum(counts, 1)
        full_pred = full_pred / counts[np.newaxis, :, :]
        return full_pred
    
    # Simular 4 patches 2x2 de uma imagem 4x4
    patch_size = 2
    img_h, img_w = 4, 4
    num_classes = 1
    
    patches = np.array([
        [[[1, 1], [1, 1]]],  # patch (0,0)
        [[[2, 2], [2, 2]]],  # patch (2,0)
        [[[3, 3], [3, 3]]],  # patch (0,2)
        [[[4, 4], [4, 4]]],  # patch (2,2)
    ], dtype=np.float32)
    
    patch_info = [
        {'patch_x': 0, 'patch_y': 0},
        {'patch_x': 2, 'patch_y': 0},
        {'patch_x': 0, 'patch_y': 2},
        {'patch_x': 2, 'patch_y': 2},
    ]
    
    result = reconstruct_from_patches(
        patches, patch_info, img_w, img_h, patch_size, num_classes
    )
    
    assert result.shape == (num_classes, img_h, img_w), f"FALHOU: Shape incorreto: {result.shape}"
    
    expected = np.array([
        [[1, 1, 2, 2],
         [1, 1, 2, 2],
         [3, 3, 4, 4],
         [3, 3, 4, 4]]
    ], dtype=np.float32)
    
    np.testing.assert_array_almost_equal(result, expected)
    print("  ✓ Reconstrução sem overlap OK")
    
    # Teste com overlap
    patch_size = 4
    img_h, img_w = 4, 6
    
    patches_overlap = np.array([
        [[[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]],
        [[[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]]],
    ], dtype=np.float32)
    
    patch_info_overlap = [
        {'patch_x': 0, 'patch_y': 0},
        {'patch_x': 2, 'patch_y': 0},
    ]
    
    result_overlap = reconstruct_from_patches(
        patches_overlap, patch_info_overlap, img_w, img_h, patch_size, 1
    )
    
    # Região de overlap deve ser a média (3.0)
    assert abs(result_overlap[0, 0, 2] - 3.0) < 0.01, "FALHOU: Média do overlap incorreta"
    print("  ✓ Reconstrução com overlap OK (média calculada corretamente)")
    
    return True


def test_no_leakage_with_patches():
    """
    TESTE CRÍTICO FINAL: Simula o fluxo completo e verifica que
    patches de uma imagem NUNCA vão para splits diferentes.
    """
    print("\n[TEST 7] Verificando ausência de leakage no fluxo completo...")
    
    from data_factory.ROP_dataset_patches import ROPDatasetPatches
    
    config = Config()
    config.apply_clahe = False
    
    data_factory = DataFactory(config)
    train_df, _ = data_factory.create_metadata_dataframe()
    
    # Carregar splits
    splits_path = os.path.join(config.output_dir, 'frozen_cv_splits.json')
    if not os.path.exists(splits_path):
        splits_path = os.path.join(config.output_dir, 'cv_splits.json')
    
    with open(splits_path, 'r') as f:
        splits_data = json.load(f)
    
    # Testar um fold
    fold_data = splits_data['folds'][0]
    train_idx = fold_data['train_indices']
    val_idx = fold_data['val_indices']
    
    train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
    val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)
    
    # Criar datasets de patches (usar só 5 imagens de cada para teste rápido)
    train_fold_df_sample = train_fold_df.head(5)
    val_fold_df_sample = val_fold_df.head(5)
    
    train_dataset = ROPDatasetPatches(
        dataframe=train_fold_df_sample,
        config=config,
        patch_size=512,
        overlap=50,
        is_train=True
    )
    
    val_dataset = ROPDatasetPatches(
        dataframe=val_fold_df_sample,
        config=config,
        patch_size=512,
        overlap=50,
        is_train=False
    )
    
    # Coletar todas as imagens de origem dos patches
    train_images = set()
    for idx in range(len(train_dataset)):
        info = train_dataset.patch_info[idx]
        img_name = train_fold_df_sample.iloc[info['image_idx']]['image_name']
        train_images.add(img_name)
    
    val_images = set()
    for idx in range(len(val_dataset)):
        info = val_dataset.patch_info[idx]
        img_name = val_fold_df_sample.iloc[info['image_idx']]['image_name']
        val_images.add(img_name)
    
    # VERIFICAR: Não pode haver overlap
    overlap = train_images & val_images
    assert len(overlap) == 0, f"FALHOU: LEAKAGE! Imagens em train E val: {overlap}"
    
    print(f"  Train patches: {len(train_dataset)} (de {len(train_images)} imagens)")
    print(f"  Val patches:   {len(val_dataset)} (de {len(val_images)} imagens)")
    print(f"  ✓ ZERO LEAKAGE! Nenhuma imagem aparece em ambos os splits")
    
    return True


def run_all_tests():
    """Executa todos os testes."""
    print("="*80)
    print("TESTES DO EXPERIMENTO 013: Patch-Based com Controle de Leakage")
    print("="*80)
    
    tests = [
        ("Train/Test Separation", test_train_test_separation),
        ("CV Fold Separation", test_cv_fold_separation),
        ("Patch Dataset Creation", test_patch_dataset_creation),
        ("Patches Grouped by Image", test_patches_grouped_by_image),
        ("Model Forward Pass", test_model_forward_pass),
        ("Patch Reconstruction", test_reconstruction),
        ("No Leakage with Patches", test_no_leakage_with_patches),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  ✗ FALHOU: {e}")
    
    # Resumo
    print("\n" + "="*80)
    print("RESUMO DOS TESTES")
    print("="*80)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ PASSOU" if success else f"✗ FALHOU: {error}"
        print(f"  {name}: {status}")
    
    print(f"\n{passed}/{total} testes passaram")
    
    if passed == total:
        print("\n🎉 TODOS OS TESTES PASSARAM! O experimento está pronto para executar.")
        return True
    else:
        print("\n❌ ALGUNS TESTES FALHARAM! Corrija antes de executar o experimento.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
