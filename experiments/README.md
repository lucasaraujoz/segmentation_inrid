# Experiments Directory

Este diretório contém todos os scripts de treinamento dos experimentos realizados.

## 📁 Organização dos Scripts

### Baseline
- `verify_baseline.py` - Script para verificar/reproduzir o baseline original

### Experimentos Arquiteturais
- `train_aspp.py` - ASPP Bottleneck (Test Dice: 0.6230, -3.30%)
- `train_aspp_decoder.py` - ASPP Decoder (Test Dice: 0.5947, -7.77%)
- `train_attention.py` - Attention Gates (Test Dice: 0.6182 fixed, -4.13%)
- `train_efficientnet.py` - Variações de EfficientNet encoder
- `train_efficientnet_b4_optimized.py` - EfficientNet-B4 otimizado

### Experimentos de Processamento de Imagem
- `train_enhanced_processing.py` - Enhanced preprocessing techniques
- `eval_morphological_postprocessing.py` - Avaliação de pós-processamento morfológico

### Experimentos de Augmentação
- `train_extreme_augmentation.py` - Augmentação extrema (Test Dice: 0.6422, -0.40%)
- `train_moderate_augmentation.py` - Augmentação moderada (Test Dice: 0.6009, -6.80%)

### Utilidades
- `evaluate_test_ensemble.py` - Script para avaliar ensemble + TTA no test set

## 🎯 Melhor Resultado

**Baseline: 0.6448 Test Dice** (Ensemble 5-fold + TTA)

Todos os experimentos falharam em superar o baseline devido ao dataset muito pequeno (54 imagens).

## 📊 Documentação Completa

Ver: `../docs/EXPERIMENTOS.md` para análise detalhada de todos os experimentos.

## 🚀 Como Usar

### Treinar um Experimento
```bash
python experiments/[script_name].py
```

### Avaliar Test Set
```bash
python experiments/evaluate_test_ensemble.py
```

### Reproduzir Baseline
```bash
python experiments/verify_baseline.py
```

## 📝 Logs

Todos os logs de treinamento estão em: `../logs/`

## ⚠️ Nota Importante

Estes scripts foram desenvolvidos sequencialmente durante a pesquisa. Alguns podem ter dependências ou configurações específicas. Para resultados reproduzíveis, usar sempre `verify_baseline.py`.
