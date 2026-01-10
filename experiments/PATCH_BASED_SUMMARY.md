# 🧩 Experimento 12: Patch-Based Segmentation

## ✅ Status: Implementado e Testado

## 📦 Arquivos Criados

### Core Implementation
1. **`data_factory/ROP_dataset_patches.py`** (270 linhas)
   - Dataset PyTorch para extração de patches
   - Sliding window com overlap configurável
   - Preserva informação espacial para reconstrução

2. **`experiments/train_patch_based.py`** (420 linhas)
   - Script de treinamento com cross-validation
   - Função `reconstruct_from_patches()` para avaliação
   - Suporte completo para test set

3. **`tests/test_patch_dataset.py`** (120 linhas)
   - Testes automatizados do dataset
   - Visualização de patches
   - Validação de dimensões

### Documentação
4. **`docs/EXPERIMENTO_12_PATCH_BASED.md`** 
   - Documentação técnica completa
   - Justificativa e metodologia
   - Resultados esperados

5. **`experiments/PATCH_BASED_README.md`**
   - Guia rápido de uso
   - Quick start
   - Exemplos práticos

6. **`docs/EXPERIMENTOS.md`** (atualizado)
   - Entrada do experimento 12
   - Integrado com histórico de experimentos

## 🧪 Testes Realizados

```bash
$ python tests/test_patch_dataset.py
```

**Resultados:**
- ✅ Dataset criado: 210 patches (3 imagens × 70 patches)
- ✅ Dimensões corretas: [3, 512, 512] para imagem
- ✅ Dimensões corretas: [2, 512, 512] para máscara
- ✅ Posicionamento: Grid 10×7 verificado
- ✅ Overlap: 50px funcionando
- ✅ Visualização: Salva em `outputs/patch_visualization.png`

## 📊 Configuração Final

```python
# Parâmetros de Patches
PATCH_SIZE = 512      # Tamanho do patch
OVERLAP = 50          # Overlap entre patches (10%)
STRIDE = 462          # Passo da sliding window

# Grid Resultante
PATCHES_WIDTH = 10    # Patches na largura (4288 ÷ 462)
PATCHES_HEIGHT = 7    # Patches na altura (2848 ÷ 462)
TOTAL_PATCHES = 70    # Por imagem

# Dataset
TRAIN_IMAGES = 54
TRAIN_PATCHES = 3,780  # 54 × 70
TEST_IMAGES = 27
TEST_PATCHES = 1,890   # 27 × 70

# Treinamento
BATCH_SIZE = 16       # Pode ser maior (patches menores)
ENCODER = 'resnet34'
EPOCHS = 50
```

## 🎯 Comparação

| Aspecto | Baseline (512×512) | Patch-Based |
|---------|-------------------|-------------|
| **Resolução** | Downsampled 8.4× | Full ✅ |
| **Amostras/época** | 54 | 3,780 ✅ |
| **Lesões pequenas** | Perdem detalhes | Preservadas ✅ |
| **Bordas** | Borradas | Nítidas ✅ |
| **Batch size** | 8-16 | 16-32 ✅ |
| **Tempo/época** | 5-10 min | 60-90 min ⚠️ |
| **Complexidade** | Simples | Reconstrução |

## 🚀 Como Executar

### 1. Verificar Implementação
```bash
python tests/test_patch_dataset.py
```

### 2. Treinar Modelo
```bash
python experiments/train_patch_based.py
```

Isso executará:
- ✅ Cross-validation com 5 folds
- ✅ ~3,780 patches por fold de treino
- ✅ Avaliação com reconstrução de imagem completa
- ✅ Salvamento de resultados em JSON

### 3. Resultados
```
outputs/
├── patch_based_results.json          # Métricas completas
├── patch_visualization.png           # Visualização de patches
└── checkpoints/
    └── patch_based/
        ├── fold_0_best.pth
        ├── fold_1_best.pth
        └── ...
```

## 📈 Resultados Esperados

### Baseline (Resize para 512×512)
```
Test Dice: 0.6448
Test IoU:  0.4775

Per-class:
  Exudates:    0.7012
  Hemorrhages: 0.5884
```

### Patch-Based (Esperado)
```
Test Dice: 0.65-0.70  (+5-10%)
Test IoU:  0.48-0.52  (+5-10%)

Melhorias esperadas em:
  ✓ Microaneurismas (lesões pequenas)
  ✓ Bordas de exudatos (mais definidas)
  ✓ Hemorragias pontuais (melhor detecção)
```

## 🔧 Próximas Variações

1. **Overlap maior (100px)**: Mais suavização
2. **Patch 640×640**: Mais contexto
3. **Patch 384×384**: Mais amostras
4. **Weighted reconstruction**: Peso no centro
5. **Patch-based + TTA**: Combinar técnicas

## 📚 Princípios Seguidos (AGENT.md)

✅ **Single Responsibility Principle**
- `ROP_dataset_patches.py`: Apenas extração de patches
- `train_patch_based.py`: Apenas treinamento e avaliação
- Sem lógica misturada

✅ **Não mistura responsabilidades**
- Dataset não faz splits
- DataFactory não carrega tensores
- TrainWorker não cria patches

✅ **Reprodutibilidade**
- Seeds configuradas
- GroupKFold mantido
- Patient_id respeitado

✅ **Clareza arquitetural**
- Código bem documentado
- Funções com docstrings
- Testes automatizados

## 🎓 Contribuições do Experimento

1. **Técnica**: Sliding window com overlap para resolução completa
2. **Implementação**: Dataset reutilizável para outros projetos
3. **Avaliação**: Reconstrução de imagem com média de overlaps
4. **Documentação**: Completa e reproduzível

## 📝 Notas Finais

- ⏱️ **Tempo**: Implementação levou ~2h (arquitetura clara ajudou)
- 🧪 **Testes**: Todos passando
- 📖 **Docs**: Completa em 3 níveis (técnica, quick start, histórico)
- 🎯 **Pronto**: Para executar e comparar com baseline

---

**Criado por**: GitHub Copilot (Claude Sonnet 4.5)  
**Data**: 2026-01-08  
**Tempo de implementação**: ~2 horas  
**Linhas de código**: ~810 linhas (core + tests)  
**Linhas de documentação**: ~600 linhas
