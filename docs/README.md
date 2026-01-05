# Documentação do Projeto

## 📚 Documentos Disponíveis

### [EXPERIMENTOS.md](EXPERIMENTOS.md)
Documentação completa de todos os experimentos realizados para melhorar a segmentação de ROP.

**Conteúdo:**
- ✅ Baseline verificado (0.6448 Test Dice)
- 🔬 10 experimentos completos com análises detalhadas
- 📊 Comparações e rankings
- 🔍 Insights técnicos profundos
- 🎯 Conclusões e recomendações

**Experimentos documentados:**
1. Baseline (EfficientNet-B4 + UNet)
2. Boundary Loss
3. ASPP Bottleneck
4. Attention Gates (Buggy)
5. Attention Gates (Fixed)
6. ASPP Decoder
7. Green Channel CLAHE
8. Morphological Post-Processing
9. Frangi Vessel Enhancement (Abandonado)
10. Extreme Augmentation
11. Moderate Augmentation

**Conclusão Principal:**
Com apenas 54 imagens de treino, o baseline simples (0.6448) é o melhor resultado. Todas as tentativas de melhoria (arquiteturas complexas, augmentação avançada, processamento de imagem) falharam devido ao dataset muito pequeno.

---

## 🎯 Resultados Principais

| Experimento | Test Dice | Δ vs Baseline | Status |
|-------------|-----------|---------------|--------|
| **Baseline** | **0.6448** | **0.00%** | ✅ **Melhor** |
| Extreme Aug | 0.6422 | -0.40% | ❌ |
| ASPP Bottleneck | 0.6230 | -3.30% | ❌ |
| Attention Gates | 0.6182 | -4.13% | ❌ |
| Moderate Aug | 0.6009 | -6.80% | ❌ |
| ASPP Decoder | 0.5947 | -7.77% | ❌ |

---

## 📖 Navegação

```
tapi_inrid/
├── docs/                          # ← VOCÊ ESTÁ AQUI
│   ├── README.md                  # Este arquivo
│   └── EXPERIMENTOS.md            # Documentação completa dos experimentos
│
├── experiments/                   # Scripts de treinamento
│   ├── README.md                  # Guia dos scripts
│   ├── verify_baseline.py         # Reproduzir baseline
│   ├── train_*.py                 # Experimentos individuais
│   └── evaluate_test_ensemble.py  # Avaliar test set
│
├── logs/                          # Logs de todos os treinamentos
│   ├── verify_baseline_*.log
│   ├── training_*.log
│   └── ...
│
├── outputs/                       # Checkpoints e resultados
│   ├── checkpoints/
│   │   ├── baseline_verify/       # Melhor modelo (0.6448)
│   │   ├── extreme_augmentation/
│   │   ├── moderate_augmentation/
│   │   └── ...
│   └── *.json                     # Resultados em JSON
│
├── configs/                       # Configurações
├── data_factory/                  # Dataset loaders
├── models/                        # Arquiteturas de modelos
├── utils/                         # Utilitários
└── notebooks/                     # Jupyter notebooks para análise
```

---

## 🚀 Quick Start

### 1. Ver Documentação dos Experimentos
```bash
cat docs/EXPERIMENTOS.md
# ou abrir no VS Code
```

### 2. Reproduzir Melhor Resultado (Baseline)
```bash
python experiments/verify_baseline.py
```

### 3. Avaliar Test Set
```bash
python experiments/evaluate_test_ensemble.py
```

### 4. Ver Logs de Treinamento
```bash
ls logs/
tail -n 50 logs/verify_baseline_with_test_EXPERIMENT1.log
```

---

## 📊 Estrutura de Checkpoints

```
outputs/checkpoints/
├── baseline_verify/              # ✅ 0.6448 (MELHOR)
│   ├── fold_0_best.pth
│   ├── fold_1_best.pth
│   ├── fold_2_best.pth
│   ├── fold_3_best.pth
│   └── fold_4_best.pth
│
├── extreme_augmentation/         # ❌ 0.6422
├── moderate_augmentation/        # ❌ 0.6009
├── aspp_bottleneck/              # ❌ 0.6230
├── attention_gates/              # ❌ 0.6182
└── aspp_decoder/                 # ❌ 0.5947
```

---

## 📝 Notas Importantes

### Dataset
- **54 imagens** de treino
- **27 imagens** de teste
- **81 pacientes** total
- GroupKFold 5-fold cross-validation (por patient_id)

### Limitações
O dataset é **muito pequeno** para:
- Arquiteturas complexas (overfitting)
- Augmentação avançada (MixUp, CutMix)
- Técnicas modernas (Transformers, etc.)

### Recomendação
**Usar o baseline** como resultado final:
- Test Dice: 0.6448
- Arquitetura: EfficientNet-B4 + UNet
- Pré-processamento: CLAHE LAB L-channel
- Ensemble: 5-fold + TTA (4 transforms)

---

## 🔗 Links Úteis

- [Experimentos Completos](EXPERIMENTOS.md)
- [Scripts](../experiments/)
- [Logs](../logs/)
- [Checkpoints](../outputs/checkpoints/)

---

**Última atualização:** Janeiro 2026  
**Melhor resultado:** Baseline 0.6448 ✅
