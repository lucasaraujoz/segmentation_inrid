# Experiment 3: Attention Gates

## Objetivo
Melhorar o baseline (Dice 0.6442) adicionando Attention Gates nas skip connections do UNet.

## Motivação Científica
- Paper: "Attention U-Net: Learning Where to Look for the Pancreas" (2018)
- Citações: 3000+
- Evidência forte em medical imaging
- Foca em regiões relevantes (lesões) e suprime fundo

## Arquitetura

### Baseline
- UNet + EfficientNet-B4
- Parâmetros: 20.3M
- Dice: 0.6442

### Attention Gates  
- UNet + EfficientNet-B4 + Attention Gates
- Parâmetros: 20.3M (+63k attention gates, +0.31%)
- Attention Gates inseridos entre encoder e decoder

## Detalhes Técnicos

### Attention Gate
```
Input:
  - x (skip): features do encoder [B, C_skip, H, W]
  - g (gate): features do decoder [B, C_gate, H', W']

Output:
  - x_att: skip connection com atenção [B, C_skip, H, W]

Operação:
  1. Transform gate: W_g(g) -> [B, C_int, H, W]
  2. Transform skip: W_x(x) -> [B, C_int, H, W]  
  3. Combine: ReLU(W_g + W_x)
  4. Attention map: Sigmoid(Conv(combined)) -> [B, 1, H, W]
  5. Apply: x * attention_map
```

### Posicionamento
```
Encoder (EfficientNet-B4):
  stage0: [B, 48, 256, 256]   -->  AG1  -->  Decoder Block 1
  stage1: [B, 32, 128, 128]   -->  AG2  -->  Decoder Block 2
  stage2: [B, 56, 64, 64]     -->  AG3  -->  Decoder Block 3
  stage3: [B, 160, 32, 32]    -->  AG4  -->  Decoder Block 4
  stage4: [B, 448, 16, 16]    -->  AG5  -->  Decoder Block 5
```

## Hipótese

Attention Gates devem melhorar porque:

1. **Foco em lesões**: Hemorrhages e exudates são pequenas
   - Baseline Hemorrhages: 0.5837 (pior classe)
   - Attention gates podem focar melhor nessas regiões

2. **Supressão de fundo**: Muito background saudável
   - Attention gates aprendem a ignorar tecido irrelevante
   - Reduz falsos positivos

3. **Poucos parâmetros**: +63k (+0.31%)
   - Menor risco de overfitting vs ASPP (+3.6M, +15%)
   - Dataset pequeno (54 imagens) favorece modificações leves

4. **Evidência forte**: Paper com 3000+ citações
   - Comprovado em segmentação médica
   - Pancreas, retina, tumores cerebrais

## Configuração de Treino

```python
model_name: "unet_attention"
encoder: "efficientnet-b4"
encoder_weights: "imagenet"
resolution: 512x512
batch_size: 8
epochs: 50
learning_rate: 1e-4
optimizer: AdamW
scheduler: ReduceLROnPlateau
loss: DiceLoss (50%) + FocalLoss (50%)
class_weights: [1.0, 1.3]  # [Exudates, Hemorrhages]
```

## Cross-Validation

- 5-fold GroupKFold
- Agrupamento por paciente (evita data leakage)
- Splits congelados (outputs/cv_splits.json)
- Hash SHA256: garantia de reprodutibilidade

## Métricas

Métricas principais:
- Dice Score (média das 2 classes)
- Per-class Dice: Exudates, Hemorrhages

Baseline para comparação:
```
Overall:      0.6442
Exudates:     0.7046
Hemorrhages:  0.5837
```

## Objetivo

**Meta: Dice > 0.6442 (+0.00% improvement mínimo)**

Razões do fracasso anterior:
1. Boundary Loss: implementação incorreta (-99.84%)
2. ASPP Bottleneck: placement errado, rates muito grandes (-3.30%)

Attention Gates é mais conservador e tem melhor evidência.

## Arquivos

- `models/unet_attention.py`: Implementação
- `test_attention.py`: Testes (PASSARAM ✅)
- `train_attention.py`: Script de treino
- `train_and_val_worker.py`: Atualizado com suporte "unet_attention"

## Como Executar

```bash
# 1. Testar implementação (já passou)
python test_attention.py

# 2. Treinar 5-fold CV
python train_attention.py

# 3. Resultados salvos em
outputs/attention_results.json
outputs/checkpoints/unet_attention_fold*.pth
```

## Próximos Passos (se falhar)

Se Attention Gates não melhorar baseline:

1. **ASPP no decoder**: rates [3,6,9] em feature maps maiores (64x64, 128x128)
2. **Deep Supervision**: loss em múltiplas escalas
3. **Ensemble**: combinar múltiplos modelos baseline
4. **Data Augmentation**: revisitar transformações

Mas expectativa é que Attention Gates funcione!
