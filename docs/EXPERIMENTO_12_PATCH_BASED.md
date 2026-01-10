# Experimento 12: Patch-Based Segmentation

## 🎯 Objetivo

Treinar o modelo usando patches extraídos das imagens originais em alta resolução (4288×2848), em vez de redimensionar para 512×512. Isso permite:
- **Preservar detalhes finos** das lesões
- **Aumentar o número de amostras** de treinamento
- **Processar imagens em resolução completa** sem limitações de memória

## 📊 Estratégia

### Extração de Patches
- **Tamanho do patch**: 512×512 pixels
- **Overlap**: 50 pixels (~10%)
- **Stride**: 462 pixels
- **Patches por imagem**: ~54 patches (9×6 grid)
- **Total de patches no treino**: ~2,916 patches (54 imagens × 54 patches)

### Grid de Patches
```
Imagem original: 4288×2848
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  1  │  2  │  3  │  4  │  5  │  6  │  7  │  8  │  9  │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ 10  │ 11  │ 12  │ 13  │ 14  │ 15  │ 16  │ 17  │ 18  │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ 19  │ 20  │ 21  │ 22  │ 23  │ 24  │ 25  │ 26  │ 27  │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ 28  │ 29  │ 30  │ 31  │ 32  │ 33  │ 34  │ 35  │ 36  │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ 37  │ 38  │ 39  │ 40  │ 41  │ 42  │ 43  │ 44  │ 45  │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ 46  │ 47  │ 48  │ 49  │ 50  │ 51  │ 52  │ 53  │ 54  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

## 🏗️ Arquitetura

### Componentes Criados

1. **`ROP_dataset_patches.py`**
   - Dataset que extrai patches das imagens originais
   - Calcula posições de todos os patches no `__init__`
   - Aplica CLAHE na imagem completa antes de extrair patches
   - Suporta sliding window com overlap configurável

2. **`train_patch_based.py`**
   - Script de treinamento principal
   - Função `reconstruct_from_patches()` para avaliar imagens completas
   - Avaliação especial no test set com reconstrução de predições

### Pipeline de Treinamento

```
1. Carregar imagem completa (4288×2848)
   ↓
2. Aplicar CLAHE
   ↓
3. Extrair patches 512×512 com overlap
   ↓
4. Aplicar augmentations (apenas nos patches)
   ↓
5. Treinar U-Net em patches individuais
```

### Pipeline de Inferência

```
1. Carregar imagem completa (4288×2848)
   ↓
2. Aplicar CLAHE
   ↓
3. Extrair patches 512×512 com overlap
   ↓
4. Predição em cada patch
   ↓
5. Reconstruir imagem completa
   ↓
6. Média das regiões overlapping
   ↓
7. Threshold (0.5) e métricas
```

## ⚙️ Configuração

```python
PATCH_SIZE = 512      # Tamanho do patch
OVERLAP = 50          # Overlap entre patches
BATCH_SIZE = 16       # Maior batch size (patches menores)
ENCODER = 'resnet34'  # Encoder baseline
EPOCHS = 50
LOSS = 'dice+focal'
```

## 📈 Vantagens Esperadas

1. **Resolução Completa**
   - Nenhuma perda de informação por downsampling
   - Lesões pequenas preservadas
   - Bordas mais nítidas

2. **Mais Dados de Treinamento**
   - 54 imagens → ~2,916 patches
   - 54× mais amostras por época
   - Melhor generalização

3. **Eficiência de Memória**
   - Batch size pode ser aumentado (patches são menores)
   - Processar imagens de qualquer tamanho

4. **Contexto Local**
   - Overlap preserva contexto nas bordas
   - Reconstrução suaviza predições

## 🔬 Reconstrução de Imagem

A função `reconstruct_from_patches()` implementa:

1. **Acumulação**: Soma predições de patches overlapping
2. **Contagem**: Registra quantas vezes cada pixel foi predito
3. **Média**: Divide pela contagem para obter média
4. **Resultado**: Predição suavizada nas regiões de overlap

```python
full_pred = sum(patches) / count(patches)
```

Isso reduz artefatos de borda e melhora a continuidade.

## 📋 Como Executar

```bash
# Executar treinamento patch-based
python experiments/train_patch_based.py
```

## 🎯 Métricas de Avaliação

### Cross-Validation
- Treino e validação em patches
- Métricas calculadas por patch

### Test Set
- **Reconstrução completa** de cada imagem
- Métricas calculadas na imagem completa (4288×2848)
- Comparação justa com ground truth

## 📊 Resultados Esperados

Comparação com baseline (resize para 512×512):

| Métrica | Baseline | Patch-Based (esperado) | Ganho |
|---------|----------|------------------------|-------|
| Dice    | 0.45-0.50| 0.50-0.55             | +5-10%|
| IoU     | 0.35-0.40| 0.40-0.45             | +5-10%|

**Principais melhorias esperadas em:**
- Microaneurismas (lesões pequenas)
- Bordas de exudatos
- Hemorragias pontuais

## 🔄 Variações Possíveis

1. **Overlap maior** (100-150px): Mais suavização
2. **Patches maiores** (640×640): Mais contexto
3. **Patches menores** (384×384): Mais amostras
4. **Weighted reconstruction**: Dar mais peso ao centro do patch
5. **Multi-scale patches**: Combinar diferentes tamanhos

## 📝 Observações

- Tempo de treinamento aumenta (~54×)
- Inferência também é mais lenta
- Requer pós-processamento de reconstrução
- Overlap ajuda na transição entre patches
- Importante manter GroupKFold para evitar data leakage

## 🚀 Próximos Passos

1. Executar experimento baseline com patches
2. Analisar qualidade das predições reconstruídas
3. Testar diferentes tamanhos de overlap
4. Comparar tempo vs qualidade
5. Avaliar em conjunto com TTA e ensemble
