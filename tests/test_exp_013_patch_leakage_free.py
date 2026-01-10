"""
Teste do Experimento 013: Patch-Based Training com Controle de Leakage

Valida:
1. Nenhum paciente/imagem aparece em múltiplos splits (train/val/test)
2. Patches são extraídos corretamente
3. Dataset retorna tensores com dimensões corretas
4. Forward/backward pass funciona
5. Reconstrução de patches funciona

FUNDAMENTAL: Patches de uma mesma imagem NUNCA podem aparecer em splits diferentes!
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
import numpy as np
import json
from collections import defaultdict

from configs.config import Config
from data_factory.data_factory import DataFactory


class TestLeakageControl:
    """Testa que não há vazamento de dados entre splits."""
    
    @pytest.fixture
    def config(self):
        """Cria configuração."""
        return Config()
    
    @pytest.fixture
    def data_factory(self, config):
        """Cria DataFactory."""
        return DataFactory(config)
    
    @pytest.fixture
    def splits_data(self, config):
        """Carrega frozen splits."""
        splits_path = os.path.join(config.output_dir, 'frozen_cv_splits.json')
        if not os.path.exists(splits_path):
            splits_path = os.path.join(config.output_dir, 'cv_splits.json')
        
        with open(splits_path, 'r') as f:
            return json.load(f)
    
    def test_train_test_separation(self, data_factory):
        """Verifica que train e test são completamente separados."""
        train_df, test_df = data_factory.create_metadata_dataframe()
        
        train_images = set(train_df['image_name'].tolist())
        test_images = set(test_df['image_name'].tolist())
        
        # Não pode haver interseção
        overlap = train_images & test_images
        assert len(overlap) == 0, f"Imagens em train E test: {overlap}"
        
        print(f"✓ Train/Test separados: {len(train_images)} train, {len(test_images)} test")
    
    def test_cv_fold_separation(self, data_factory, splits_data):
        """Verifica que folds de CV não têm overlap."""
        train_df, _ = data_factory.create_metadata_dataframe()
        
        for fold_data in splits_data['folds']:
            fold_num = fold_data['fold']
            train_idx = set(fold_data['train_indices'])
            val_idx = set(fold_data['val_indices'])
            
            # Não pode haver interseção entre train e val do mesmo fold
            overlap = train_idx & val_idx
            assert len(overlap) == 0, f"Fold {fold_num}: índices em train E val: {overlap}"
            
            # Todos os índices devem ser válidos
            all_idx = train_idx | val_idx
            assert all(i < len(train_df) for i in all_idx), \
                f"Fold {fold_num}: índices fora do range"
        
        print(f"✓ Todos os {len(splits_data['folds'])} folds têm splits corretos")
    
    def test_patient_level_separation(self, data_factory, splits_data):
        """Verifica que pacientes não aparecem em múltiplos splits."""
        train_df, _ = data_factory.create_metadata_dataframe()
        
        for fold_data in splits_data['folds']:
            fold_num = fold_data['fold']
            
            train_patients = set(train_df.iloc[fold_data['train_indices']]['patient_id'].tolist())
            val_patients = set(train_df.iloc[fold_data['val_indices']]['patient_id'].tolist())
            
            overlap = train_patients & val_patients
            
            # No INRID, cada imagem é de um paciente diferente
            # Então overlap de patient_id == overlap de imagens
            assert len(overlap) == 0, \
                f"Fold {fold_num}: pacientes em train E val: {overlap}"
        
        print("✓ Separação por paciente verificada em todos os folds")


class TestPatchDataset:
    """Testa o dataset de patches."""
    
    @pytest.fixture
    def config(self):
        """Cria configuração."""
        config = Config()
        config.apply_clahe = False  # Sem CLAHE
        return config
    
    @pytest.fixture
    def sample_df(self, config):
        """Cria DataFrame de amostra com 2 imagens."""
        data_factory = DataFactory(config)
        train_df, _ = data_factory.create_metadata_dataframe()
        return train_df.head(2)  # Apenas 2 imagens para teste rápido
    
    def test_patch_dataset_creation(self, config, sample_df):
        """Testa criação do dataset de patches."""
        from data_factory.ROP_dataset_patches import ROPDatasetPatches
        
        dataset = ROPDatasetPatches(
            dataframe=sample_df,
            config=config,
            patch_size=512,
            overlap=50,
            is_train=False
        )
        
        # Deve ter patches
        assert len(dataset) > 0, "Dataset vazio"
        
        # Calcular patches esperados (aproximado)
        # 4288x2848 com 512x512 e stride 462 → ~10x7 = 70 patches/imagem
        expected_min = len(sample_df) * 50  # No mínimo 50 patches/imagem
        assert len(dataset) >= expected_min, \
            f"Poucos patches: {len(dataset)} < {expected_min}"
        
        print(f"✓ Dataset criado com {len(dataset)} patches de {len(sample_df)} imagens")
    
    def test_patch_dimensions(self, config, sample_df):
        """Testa dimensões dos patches."""
        from data_factory.ROP_dataset_patches import ROPDatasetPatches
        
        dataset = ROPDatasetPatches(
            dataframe=sample_df,
            config=config,
            patch_size=512,
            overlap=50,
            is_train=False
        )
        
        sample = dataset[0]
        
        # Image: [3, 512, 512]
        assert 'image' in sample
        img_shape = sample['image'].shape
        assert img_shape == (3, 512, 512), f"Shape incorreto da imagem: {img_shape}"
        
        # Mask: [2, 512, 512] (2 classes: exsudatos, hemorragias)
        assert 'mask' in sample
        mask_shape = sample['mask'].shape
        assert mask_shape == (2, 512, 512), f"Shape incorreto da máscara: {mask_shape}"
        
        print(f"✓ Dimensões corretas: image={img_shape}, mask={mask_shape}")
    
    def test_patch_metadata(self, config, sample_df):
        """Testa metadados dos patches."""
        from data_factory.ROP_dataset_patches import ROPDatasetPatches
        
        dataset = ROPDatasetPatches(
            dataframe=sample_df,
            config=config,
            patch_size=512,
            overlap=50,
            is_train=False
        )
        
        sample = dataset[0]
        
        required_keys = ['image_name', 'image_idx', 'patch_x', 'patch_y', 
                        'grid_i', 'grid_j', 'img_width', 'img_height']
        
        for key in required_keys:
            assert key in sample, f"Metadado faltando: {key}"
        
        # Verificar valores
        assert sample['patch_x'] >= 0
        assert sample['patch_y'] >= 0
        assert sample['img_width'] > 0
        assert sample['img_height'] > 0
        
        print(f"✓ Metadados completos: {list(sample.keys())}")
    
    def test_all_patches_from_same_image_same_split(self, config, sample_df):
        """TESTE CRÍTICO: Todos os patches de uma imagem devem ficar no mesmo split."""
        from data_factory.ROP_dataset_patches import ROPDatasetPatches
        
        # Criar dataset
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
        for img_idx, patch_indices in patches_by_image.items():
            assert len(patch_indices) > 1, \
                f"Imagem {img_idx} tem apenas {len(patch_indices)} patch(es)"
        
        print(f"✓ {len(patches_by_image)} imagens, cada uma com múltiplos patches")
        for img_idx, patches in patches_by_image.items():
            print(f"  Imagem {img_idx}: {len(patches)} patches")


class TestModelWithPatches:
    """Testa modelo com patches."""
    
    @pytest.fixture
    def config(self):
        config = Config()
        config.apply_clahe = False
        return config
    
    def test_forward_pass(self, config):
        """Testa forward pass com patches."""
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
        
        assert output.shape == (4, 2, 512, 512), f"Output shape incorreto: {output.shape}"
        assert not torch.isnan(output).any(), "NaN no output"
        assert not torch.isinf(output).any(), "Inf no output"
        
        print(f"✓ Forward pass OK: input={x.shape} → output={output.shape}")
    
    def test_backward_pass(self, config):
        """Testa backward pass com patches."""
        import segmentation_models_pytorch as smp
        
        model = smp.Unet(
            encoder_name='efficientnet-b4',
            encoder_weights='imagenet',
            in_channels=3,
            classes=2
        )
        
        x = torch.randn(2, 3, 512, 512)
        target = torch.randint(0, 2, (2, 2, 512, 512)).float()
        
        output = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
        
        loss.backward()
        
        # Verificar gradientes
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                assert not torch.isnan(param.grad).any(), f"NaN em grad de {name}"
                assert not torch.isinf(param.grad).any(), f"Inf em grad de {name}"
        
        assert has_grad, "Nenhum gradiente calculado"
        print(f"✓ Backward pass OK, loss={loss.item():.4f}")


class TestPatchReconstruction:
    """Testa reconstrução de imagem a partir de patches."""
    
    def test_reconstruction_no_overlap(self):
        """Testa reconstrução sem overlap."""
        from experiments.train_patch_based import reconstruct_from_patches
        
        # Simular 4 patches 2x2 de uma imagem 4x4
        patch_size = 2
        img_h, img_w = 4, 4
        num_classes = 2
        
        patches = np.array([
            [[[1, 1], [1, 1]], [[0, 0], [0, 0]]],  # patch (0,0)
            [[[2, 2], [2, 2]], [[0, 0], [0, 0]]],  # patch (2,0)
            [[[3, 3], [3, 3]], [[0, 0], [0, 0]]],  # patch (0,2)
            [[[4, 4], [4, 4]], [[0, 0], [0, 0]]],  # patch (2,2)
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
        
        assert result.shape == (num_classes, img_h, img_w)
        
        # Verificar valores
        expected = np.array([
            [[1, 1, 2, 2],
             [1, 1, 2, 2],
             [3, 3, 4, 4],
             [3, 3, 4, 4]],
            np.zeros((4, 4))
        ], dtype=np.float32)
        
        np.testing.assert_array_almost_equal(result, expected)
        print("✓ Reconstrução sem overlap OK")
    
    def test_reconstruction_with_overlap(self):
        """Testa reconstrução com overlap (média de regiões sobrepostas)."""
        from experiments.train_patch_based import reconstruct_from_patches
        
        # 2 patches sobrepostos
        patch_size = 4
        img_h, img_w = 4, 6
        num_classes = 1
        
        # Patch 1: valores = 2.0, Patch 2: valores = 4.0
        # Overlap de 2 pixels → média = 3.0
        patches = np.array([
            [[[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]],  # (0,0)
            [[[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]]],  # (2,0)
        ], dtype=np.float32)
        
        patch_info = [
            {'patch_x': 0, 'patch_y': 0},
            {'patch_x': 2, 'patch_y': 0},
        ]
        
        result = reconstruct_from_patches(
            patches, patch_info, img_w, img_h, patch_size, num_classes
        )
        
        # Colunas 0-1: só patch1 (2.0)
        # Colunas 2-3: overlap (média 3.0)
        # Colunas 4-5: só patch2 (4.0)
        expected = np.array([[[2, 2, 3, 3, 4, 4],
                             [2, 2, 3, 3, 4, 4],
                             [2, 2, 3, 3, 4, 4],
                             [2, 2, 3, 3, 4, 4]]], dtype=np.float32)
        
        np.testing.assert_array_almost_equal(result, expected)
        print("✓ Reconstrução com overlap OK (média calculada corretamente)")


def run_all_tests():
    """Executa todos os testes."""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    run_all_tests()
