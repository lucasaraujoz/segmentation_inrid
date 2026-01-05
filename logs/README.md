# Training Logs Archive

Este diretório contém todos os logs de treinamento dos experimentos.

## 📁 Organização

### Logs por Experimento

#### Baseline
- `verify_baseline_EXPERIMENT1.log`
- `verify_baseline_with_test_EXPERIMENT1.log`

#### Experimentos Arquiteturais
- `training_aspp_EXPERIMENT2.log` - ASPP Bottleneck
- `training_aspp_decoder_FINAL.log` - ASPP Decoder
- `training_attention_gate_EXPERIMENT1.log` - Attention Gates
- `training_boundary_loss_EXPERIMENT1.log` - Boundary Loss

#### Experimentos de Processamento
- `training_GREEN_CHANNEL.log`
- `GREEN_CHANNEL_5FOLD_CV.log`
- `test_green_clahe_fold1.log`
- `training_enhanced_processing_EXPERIMENT4.log`
- `training_enhanced_processing_EXPERIMENT4_FINAL.log`
- `training_enhanced_processing_EXPERIMENT4_v2.log`
- `training_enhanced_processing_EXPERIMENT4_v3.log`
- `training_enhanced_processing_green_channel_FINAL.log`

#### Pós-Processamento Morfológico
- `morphological_postprocessing_CORRECT.log`
- `morphological_postprocessing_FINAL.log`
- `morphological_postprocessing_results.log`

#### Augmentação
- `training_extreme_augmentation.log`
- `training_extreme_augmentation_RETRY.log`
- `training_moderate_augmentation.log`

#### EfficientNet Variações
- `training_efficientnet_b4.log`
- `training_efficientnet_b4_512_FINAL.log`
- `training_effb4_optimized.log`

#### Outros
- `training-1.0.log`
- `training_unetpp_768.log`
- `test_ensemble_results.log`

## 📊 Como Analisar Logs

### Ver últimas linhas (resumo de resultados)
```bash
tail -n 50 logs/verify_baseline_with_test_EXPERIMENT1.log
```

### Buscar métricas específicas
```bash
grep "Test Dice" logs/*.log
grep "Best Dice" logs/*.log
grep "Cross-Validation" logs/*.log
```

### Ver progresso de treinamento
```bash
grep "Epoch" logs/training_extreme_augmentation.log
```

### Contar épocas treinadas
```bash
grep -c "Epoch" logs/training_moderate_augmentation.log
```

## 🔍 Resultados Principais

| Experimento | Log | Test Dice |
|-------------|-----|-----------|
| Baseline | verify_baseline_with_test_EXPERIMENT1.log | 0.6448 |
| Extreme Aug | training_extreme_augmentation.log | 0.6422 |
| Moderate Aug | training_moderate_augmentation.log | 0.6009 |
| ASPP Bottleneck | training_aspp_EXPERIMENT2.log | 0.6230 |
| ASPP Decoder | training_aspp_decoder_FINAL.log | 0.5947 |
| Attention Gates | training_attention_gate_EXPERIMENT1.log | 0.6182 |

## 📝 Formato dos Logs

Os logs geralmente contém:
1. **Configuração:** Hiperparâmetros, arquitetura, dataset
2. **Treinamento:** Progresso epoch por epoch (train/val loss, dice, IoU)
3. **Cross-Validation:** Resultados de cada fold
4. **Test Set:** Métricas finais (Dice, IoU, per-class)
5. **Ensemble:** Resultados com ensemble + TTA

## ⚠️ Notas

- Logs com sufixo `_FINAL` são as versões definitivas dos experimentos
- Logs com `_EXPERIMENT[N]` indicam número da tentativa
- Logs com `_v2`, `_v3` são iterações/correções
- Alguns experimentos têm múltiplos logs devido a re-runs ou debugging

## 🔗 Documentação

Ver documentação completa em: `../docs/EXPERIMENTOS.md`
