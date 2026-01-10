# Arquitetura Completa do Modelo - Explicação Detalhada

## 📋 Visão Geral

**Modelo Base:** U-Net  
**Encoder:** EfficientNet-B4 (pré-treinado ImageNet)  
**Decoder:** Blocos de upsampling com skip connections  
**Modificação:** Wavelet DWT 2D no primeiro skip connection  
**Saída:** 2 canais (Exsudatos + Hemorragias)  
**Função de Ativação:** **Sigmoid** (não Softmax!)  

---

## 🏗️ 1. Estrutura Geral (U-Net)

```
INPUT (3, 512, 512)                                    OUTPUT (2, 512, 512)
      |                                                         ↑
      ↓                                                         |
  ┌─────────┐                                             ┌─────────┐
  │ ENCODER │ ──────── Skip 0 (3, 512, 512) ───────────→ │ DECODER │
  │         │                                             │         │
  │ Efficient│──────── Skip 1 (48, 256, 256) + WAVELET →│  Blocks │
  │ Net-B4  │                                             │         │
  │         │──────── Skip 2 (56, 128, 128) ────────────→│         │
  │         │                                             │         │
  │         │──────── Skip 3 (160, 64, 64) ─────────────→│         │
  │         │                                             │         │
  │         │──────── Skip 4 (272, 32, 32) ─────────────→│         │
  └─────────┘                                             └─────────┘
       |                                                       ↑
       └─── Bottleneck (448, 16, 16) ────────────────────────┘
```

**Pontos-chave:**
- Encoder **reduz** resolução espacial e **aumenta** número de canais
- Decoder **aumenta** resolução espacial e **reduz** número de canais
- Skip connections permitem que detalhes de alta resolução fluam diretamente para o decoder
- **WAVELET atua APENAS no Skip 1** (primeiro skip após entrada)

---

## 🔬 2. Encoder Detalhado (EfficientNet-B4)

O EfficientNet-B4 é dividido em **5 features** extraídas em diferentes profundidades:

```python
# Saídas do encoder em cada nível:
features = [
    features[0]: (3, 512, 512)    # Input original (RGB)
    features[1]: (48, 256, 256)   # ← WAVELET APLICADO AQUI!
    features[2]: (56, 128, 128)   
    features[3]: (160, 64, 64)    
    features[4]: (272, 32, 32)    
]

# Bottleneck (saída final do encoder):
bottleneck: (448, 16, 16)
```

### Como funciona o EfficientNet-B4:

1. **Conv Stem:** (3, 512, 512) → (48, 256, 256)
   - Primeira convolução, reduz resolução pela metade

2. **MBConv Blocks (Mobile Inverted Bottleneck):**
   - Sequência de blocos que aplicam:
     - **Expand:** Aumenta canais temporariamente
     - **Depthwise Conv:** Convolução eficiente por canal
     - **Squeeze-Excitation (SE):** Atenção nos canais
     - **Project:** Reduz canais de volta
   
3. **Progressive Downsampling:**
   - A cada bloco, reduz resolução espacial
   - Aumenta número de canais (extrai features mais complexas)

---

## 🌊 3. Wavelet Enhancement no Skip 1

**Localização:** Entre `features[1]` do encoder e decoder  
**Input:** (48, 256, 256) - primeira feature após downsampling  
**Output:** (48, 256, 256) - mesma dimensão, mas enriquecida com edges

### Processo Detalhado:

```python
# 1. Input original do skip
skip1_original = encoder.features[1]  # [B, 48, 256, 256]

# 2. Aplicar DWT 2D (Discrete Wavelet Transform - Haar)
for cada canal (48 canais):
    LL, LH, HL, HH = pywt.dwt2(canal, 'haar')
    # LL: Low-Low (aproximação) - 128×128 - DESCARTADO
    # LH: Low-High (bordas horizontais) - 128×128 - USADO
    # HL: High-Low (bordas verticais) - 128×128 - USADO
    # HH: High-High (bordas diagonais) - 128×128 - USADO

# 3. Upsample wavelets de volta para 256×256
LH_upsampled = F.interpolate(LH, size=(256, 256))  # [B, 48, 256, 256]
HL_upsampled = F.interpolate(HL, size=(256, 256))  # [B, 48, 256, 256]
HH_upsampled = F.interpolate(HH, size=(256, 256))  # [B, 48, 256, 256]

# 4. Concatenar com skip original
concatenated = torch.cat([
    skip1_original,  # [B, 48, 256, 256]
    LH_upsampled,    # [B, 48, 256, 256]
    HL_upsampled,    # [B, 48, 256, 256]
    HH_upsampled     # [B, 48, 256, 256]
], dim=1)  # Resultado: [B, 192, 256, 256]

# 5. Reduzir canais de volta para 48 (Conv 1×1 + BN + ReLU)
skip1_enhanced = WaveletModule(concatenated)  # [B, 48, 256, 256]
```

### Por que funciona?

- **LH, HL, HH** capturam **bordas em diferentes direções**
- Exsudatos e hemorragias têm **bordas bem definidas**
- Wavelet extrai essas informações de alta frequência
- Modelo consegue detectar melhor lesões pequenas e detalhadas

---

## 🔄 4. Decoder e Skip Connections

### Como as skip connections se unem:

```python
# Decoder Block (exemplo simplificado)
def decoder_block(decoder_input, skip_connection):
    # 1. Upsample decoder (dobra resolução)
    up = upsample(decoder_input)  # Ex: (128, 32, 32) → (128, 64, 64)
    
    # 2. Concatenar com skip connection
    concat = torch.cat([up, skip_connection], dim=1)  # Soma canais
    
    # 3. Convoluções para processar
    out = conv1(concat)
    out = bn1(out)
    out = relu(out)
    out = conv2(out)
    out = bn2(out)
    out = relu(out)
    
    return out
```

### Fluxo Completo do Decoder:

```
Bottleneck (448, 16, 16)
    ↓ upsample + concat skip4
Block 4: (272, 32, 32) → (160, 32, 32)
    ↓ upsample + concat skip3
Block 3: (160, 64, 64) → (56, 64, 64)
    ↓ upsample + concat skip2
Block 2: (56, 128, 128) → (48, 128, 128)
    ↓ upsample + concat skip1 (COM WAVELET!)
Block 1: (48, 256, 256) → (16, 256, 256)
    ↓ upsample + concat skip0
Block 0: (3, 512, 512) → (16, 512, 512)
    ↓
Segmentation Head
```

---

## 🎯 5. Saída e Ativação (CRITICAL!)

### Segmentation Head:

```python
# Última camada
self.segmentation_head = nn.Conv2d(
    in_channels=16,
    out_channels=2,  # ← 2 CLASSES (Exsudatos, Hemorragias)
    kernel_size=3,
    padding=1
)

# Forward
logits = self.segmentation_head(decoder_output)  # [B, 2, 512, 512]
```

### ⚠️ SIGMOID vs SOFTMAX - Diferença CRUCIAL:

**Não usamos Softmax! Usamos SIGMOID!**

```python
# Durante predição:
probs = torch.sigmoid(logits)  # [B, 2, 512, 512]

# Canal 0: Probabilidade de Exsudato (independente)
# Canal 1: Probabilidade de Hemorragia (independente)
```

### Por que SIGMOID e não SOFTMAX?

**Com Softmax (ERRADO para este caso):**
```python
probs = F.softmax(logits, dim=1)
# Se P(exsudato) = 0.8 → P(hemorragia) = 0.2
# Classes são MUTUAMENTE EXCLUSIVAS
# Um pixel SÓ pode ser exsudato OU hemorragia
```

**Com Sigmoid (CORRETO!):**
```python
probs = torch.sigmoid(logits)
# P(exsudato) = 0.8 e P(hemorragia) = 0.7 é VÁLIDO!
# Classes são INDEPENDENTES
# Um pixel PODE ter exsudato E hemorragia simultaneamente
```

### Por que precisamos dessa independência?

1. **Lesões podem se sobrepor** na imagem
2. **Background** é implícito: `P(background) = (1 - P(exsudato)) * (1 - P(hemorragia))`
3. **Multi-label classification**: Cada classe é binária independente

---

## 📊 6. Loss Function

```python
# Usamos Binary Cross-Entropy (BCE) + Dice Loss
# NÃO usamos Categorical Cross-Entropy (que exigiria Softmax)

# Para cada classe:
bce_loss = -[y * log(σ(x)) + (1-y) * log(1-σ(x))]

# Dice Loss (para cada classe separadamente):
dice_loss = 1 - (2 * |X ∩ Y|) / (|X| + |Y|)

# Total:
total_loss = α * bce_loss + β * dice_loss
```

---

## 🔍 7. Fluxo Completo de Dados (Passo a Passo)

### Input: Imagem RGB (512×512)

```python
# 1. Entrada
image = (3, 512, 512)  # RGB normalizada

# 2. Encoder (EfficientNet-B4)
encoder_out = encoder(image)
# features[0] = (3, 512, 512)
# features[1] = (48, 256, 256)  ← Será modificado!
# features[2] = (56, 128, 128)
# features[3] = (160, 64, 64)
# features[4] = (272, 32, 32)

# 3. HOOK: Modificar features[1] com Wavelet
original_skip1 = features[1]  # (48, 256, 256)
enhanced_skip1 = wavelet_module(original_skip1)  # (48, 256, 256) + edges
features[1] = enhanced_skip1  # Substitui!

# 4. Decoder (com skips modificadas)
x = bottleneck  # (448, 16, 16)

# Decoder Block 4
x = upsample(x)  # (448, 32, 32)
x = concat(x, features[4])  # (448 + 272, 32, 32) = (720, 32, 32)
x = conv_blocks(x)  # (160, 32, 32)

# Decoder Block 3
x = upsample(x)  # (160, 64, 64)
x = concat(x, features[3])  # (160 + 160, 64, 64) = (320, 64, 64)
x = conv_blocks(x)  # (56, 64, 64)

# Decoder Block 2
x = upsample(x)  # (56, 128, 128)
x = concat(x, features[2])  # (56 + 56, 128, 128) = (112, 128, 128)
x = conv_blocks(x)  # (48, 128, 128)

# Decoder Block 1 - USA SKIP COM WAVELET!
x = upsample(x)  # (48, 256, 256)
x = concat(x, features[1])  # (48 + 48, 256, 256) = (96, 256, 256)
x = conv_blocks(x)  # (16, 256, 256)

# Decoder Block 0
x = upsample(x)  # (16, 512, 512)
x = concat(x, features[0])  # (16 + 3, 512, 512) = (19, 512, 512)
x = conv_blocks(x)  # (16, 512, 512)

# 5. Segmentation Head
logits = segmentation_head(x)  # (2, 512, 512)

# 6. Ativação (SIGMOID!)
probs = torch.sigmoid(logits)  # (2, 512, 512)
# probs[0] = Probabilidade de Exsudato para cada pixel
# probs[1] = Probabilidade de Hemorragia para cada pixel
```

---

## 📈 8. Pós-Processamento e Métricas

### Binarização:

```python
threshold = 0.5
pred_exsudatos = (probs[0] > threshold).float()  # (512, 512) - binário
pred_hemorragias = (probs[1] > threshold).float()  # (512, 512) - binário
```

### Cálculo de Dice:

```python
# Para cada classe separadamente:
def dice_score(pred, target):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2 * intersection) / (union + 1e-8)
    return dice

dice_exsudatos = dice_score(pred_exsudatos, gt_exsudatos)
dice_hemorragias = dice_score(pred_hemorragias, gt_hemorragias)
dice_mean = (dice_exsudatos + dice_hemorragias) / 2
```

---

## 🎓 9. Resumo Conceitual

### Analogia com uma fábrica:

1. **Encoder (EfficientNet-B4):** Linha de montagem que **extrai features**
   - Início: Imagem simples (3 cores)
   - Fim: Representação complexa (448 features)

2. **Skip Connections:** Tubos laterais que **preservam detalhes**
   - Permitem que informação de alta resolução "pule" etapas
   - Essencial para reconstrução precisa

3. **Wavelet no Skip 1:** Departamento de **controle de qualidade de bordas**
   - Detecta bordas finas e detalhes
   - Melhora detecção de lesões pequenas

4. **Decoder:** Linha de reconstrução que **reconstrói a imagem**
   - Combina features profundas (semântica) com detalhes (skips)
   - Gera mapa de probabilidades

5. **Sigmoid:** Decisão final **independente para cada classe**
   - Não força competição entre classes
   - Permite sobreposição de lesões

---

## 📊 10. Comparação: Baseline vs Wavelet

### Baseline (sem CLAHE):

```
Input → Encoder → Skips (normais) → Decoder → Output
                                      ↓
                                  Dice: 0.6501
```

### Com Wavelet Skip 1:

```
Input → Encoder → Skip1 + Wavelet → Decoder → Output
                   ↑                   ↓
              Extrai bordas        Dice: 0.6721 (+3.4%)
```

### Por que Skip 1 especificamente?

- **Alta resolução** (256×256): Detalhes preservados
- **Não muito profundo**: Ainda captura informação espacial
- **Não muito raso**: Já tem features abstraídas
- **Equilíbrio perfeito** entre semântica e detalhes espaciais

---

## 🔧 11. Implementação Técnica (Hooks)

### Como integramos o Wavelet sem modificar o encoder?

```python
class UnetWaveletSkip1(nn.Module):
    def __init__(self, ...):
        self.base_model = smp.Unet(...)  # UNet original
        self.wavelet_skip1 = WaveletSkipConnection(48)  # Módulo wavelet
        self.register_hooks()  # Registra hook
    
    def register_hooks(self):
        # Hook intercepta saída do encoder
        self.base_model.encoder.register_forward_hook(
            self._custom_forward_hook
        )
    
    def _custom_forward_hook(self, module, input, output):
        # output = [features[0], features[1], ..., features[4]]
        features = list(output)
        
        # Modifica apenas features[1]
        features[1] = self.wavelet_skip1(features[1])
        
        return tuple(features)  # Retorna modificado
```

**Vantagem:** Não precisamos reescrever o encoder inteiro!

---

## 📝 12. Parâmetros do Modelo

```python
Total de parâmetros: 19,310,994
  - Encoder (EfficientNet-B4): ~15M parâmetros
  - Decoder: ~4M parâmetros
  - Wavelet Skip: 9,312 parâmetros (+0.05%)
  - Segmentation Head: ~300 parâmetros
```

**Overhead do Wavelet:** Praticamente zero!

---

## 🎯 13. Por que isso funciona?

1. **EfficientNet-B4:** Encoder robusto, pré-treinado, eficiente
2. **U-Net:** Arquitetura comprovada para segmentação médica
3. **Wavelet:** Extrai informação de alta frequência (bordas, texturas)
4. **Skip 1:** Localização ideal para enriquecimento de detalhes
5. **Sigmoid:** Permite multi-label (lesões podem coexistir)
6. **Dice + BCE Loss:** Lida bem com classes desbalanceadas

---

## 🚀 14. Próximos Passos Possíveis

1. **Wavelet em múltiplos skips** (Skip 1 + Skip 2)
2. **Diferentes wavelets** (Daubechies, Biorthogonal)
3. **Attention mechanisms** nos skips
4. **Multi-scale inference** (TTA melhorado)
5. **Ensemble com outros encoders** (ResNet, DenseNet)

---

## 📚 Referências

- **U-Net:** Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- **EfficientNet:** Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs"
- **Wavelet:** Mallat, "A Wavelet Tour of Signal Processing"
- **Segmentation Models PyTorch:** https://github.com/qubvel/segmentation_models.pytorch

---

**Última atualização:** 2026-01-05  
**Performance:** Test Dice = **0.6721** (Exsudatos: 0.7275, Hemorragias: 0.6167)
