# FGADR Experiments

Experimentos de segmentação multi-classe no dataset FGADR.

## Dataset

- **Nome:** FGADR (Fundus Grading for Automated Diabetic Retinopathy)
- **Tamanho:** 1842 imagens
- **Pacientes:** 1842 pacientes únicos (1 imagem por paciente - simplifica splits!)
- **Classes:** 2 classes multi-label (segmentação multi-canal)
  1. **Exudates** - Combinação de Hard Exudate + Soft Exudate
  2. **Hemorrhage** - Hemorragias

**Por quê apenas 2 classes?**
- Papers de referência focam nessas 2 classes principais
- Mais fácil comparar com benchmarks da literatura
- Classes co-ocorrem em 50% dos casos (multi-label permite isso)
- Ativação Sigmoid (não Softmax) - pixel pode ter ambas classes

## Preprocessing

Baseado em análise do notebook `fgadr_comprehensive_analysis.ipynb`:

**Pipeline Recomendado: CLAHE + Bilateral Filter**

1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
   - Realça contraste adaptativo
   - `clipLimit=2.0, tileGridSize=(8,8)`
   - Aplicado no canal L (espaço LAB)

2. **Bilateral Filter**
   - Reduz ruído preservando bordas
   - `d=9, sigmaColor=75, sigmaSpace=75`
   - Crítico para manter detalhes de lesões pequenas (MA, hemorragias)

**Por quê?**
- ✅ Melhor equilíbrio contraste/nitidez/ruído
- ✅ Contraste: 54.0 (alto)
- ✅ Nitidez: 465 (preserva bordas)
- ✅ Ruído: 11.9 (controlado)

## Arquitetura

- **Modelo:** U-Net
- **Encoder:** EfficientNet-B4 (ImageNet pre-trained)
- **Saída:** 6 canais (multi-label binary segmentation)
- **Loss:** Combined BCE + Dice Loss
- **Resolução:** 512x512

## Experimentos

### 1. `train_multiclass_segmentation.py`

Baseline para segmentação multi-classe FGADR.

**Configuração:**
```python
- Batch size: 8
- Epochs: 50
- Learning rate: 1e-4
- Train/Val split: 80/20 (1473/369 imagens)
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau
```

**Como executar:**
```bash
cd /home/lucas/mestrado/tapi_inrid
python experiments_fgadr/train_multiclass_segmentation.py
```

**Outputs:**
- `outputs/fgadr_multiclass/checkpoints/best_model.pth`
- `outputs/fgadr_multiclass/training_history.json`
- `outputs/fgadr_multiclass/config.json`

## Benchmarks do Artigo

Segundo o artigo original do FGADR, os melhores resultados foram:
- **Hemorrhage:** Dice ~0.70
- **Exudates:** Dice ~0.70

**Meta:** Superar esses resultados com U-Net + EfficientNet-B4 + CLAHE+Bilateral!

## Dataset vs IDRiD

| Métrica | IDRiD | FGADR | Diferença |
|---------|-------|-------|-----------|
| Imagens | 54 | 1842 | **34x maior** |
| Pacientes | 54 | 1842 | **34x maior** |
| Classes | 2 (EX, HE) | 2 (EX, HE) | **Mesmas!** |
| Split | Por paciente | Aleatório | Mais simples |

## Próximos Passos

1. ✅ Dataset 2-class (Exudates + Hemorrhage)
2. ✅ Baseline multi-label segmentation
3. ⏳ Análise por classe (qual classe tem pior Dice?)
4. ⏳ Transfer learning IDRiD → FGADR
5. ⏳ Ensemble de modelos
6. ⏳ Test Time Augmentation (TTA)

## Notas

- Todos os pacientes são únicos (1 imagem/paciente)
- Não precisa de split estratificado por paciente
- Co-ocorrência alta: Hemorrhage + Exudates (~50% das hemorragias)
- Multi-label com Sigmoid permite co-ocorrência
