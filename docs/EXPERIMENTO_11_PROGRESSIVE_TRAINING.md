# Experimento 11: Treinamento em 3 Fases (Freezing/Unfreezing)

## 🎯 Hipótese
Treinamento progressivo (congelar/descongelar encoder) reduz overfitting e preserva features do ImageNet, crítico para dataset pequeno (54 imagens).

## 📝 Configuração Detalhada

### Fase 1 — Warm-up do Decoder
**Objetivo:** Treinar decoder sem destruir features pré-treinadas do encoder

```python
# Encoder
encoder.requires_grad = False  # 100% congelado

# Decoder  
decoder.requires_grad = True   # Treinável

# Treinamento
epochs = 25
optimizer = AdamW(decoder.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
early_stopping = EarlyStopping(patience=8, mode='max')
```

**Justificativa:**
- Decoder aprende a usar features do encoder sem alterá-las
- 25 épocas + early stopping (patience=8) previne overfit
- LR 1e-3: agressivo mas ok (só decoder aprende)

---

### Fase 2 — Fine-tuning Parcial
**Objetivo:** Adaptar últimas camadas do encoder ao domínio específico

```python
# Encoder
# Descongelar apenas último bloco (block 6 no EfficientNet-B4)
for param in encoder.blocks[:-1].parameters():
    param.requires_grad = False
for param in encoder.blocks[-1].parameters():
    param.requires_grad = True

# Decoder
decoder.requires_grad = True

# Treinamento
epochs = 15

# OPÇÃO A: Discriminative Learning Rates (Recomendado)
optimizer = AdamW([
    {'params': encoder.blocks[-1].parameters(), 'lr': 1e-4},
    {'params': decoder.parameters(), 'lr': 5e-4}
], weight_decay=0.01)

# OPÇÃO B: LR única (mais simples, se Opção A der problema)
# optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
early_stopping = EarlyStopping(patience=5, mode='max')
```

**Justificativa:**
- Último bloco do encoder se adapta ao domínio ROP
- LRs diferentes: encoder mais conservador (1e-4), decoder mais agressivo (5e-4)
- Early stopping patience=5: deixa explorar mais

---

### Fase 3 — Refinamento Global
**Objetivo:** Fine-tuning completo com LR muito baixa

```python
# Encoder
encoder.requires_grad = True  # Totalmente descongelado

# Decoder
decoder.requires_grad = True

# Treinamento
epochs = 10
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
early_stopping = EarlyStopping(patience=5, mode='max')
```

**Justificativa:**
- LR 5e-5 (não 1e-5): conservador mas efetivo
- 10 épocas fixas: refinamento final
- Early stopping patience=5: previne overfitting

---

## 🔄 Configuração Baseline Mantida

**Pré-processamento:**
- CLAHE no canal L (LAB)
- Normalização ImageNet

**Augmentação:**
- Geométrica básica (flip, rotate, shift, scale)
- Probabilidades moderadas (0.3-0.5)
- **SEM** MixUp, CutMix

**Loss:**
- Dice Loss + BCE (α=0.5)
- Focal Loss γ=2.0
- Class weights [1.0, 1.0] (balanceado)

**Outros:**
- Batch size: 4 (ou 8 se couber)
- Image size: 512x512
- Cross-Validation: 5-fold (splits de `outputs/cv_splits.json`)

---

## 📊 Protocolo de Avaliação

### Durante Treinamento (CV)
- Salvar best model de cada fold (Val Dice)
- Monitorar Val Dice e Val Loss
- Aplicar early stopping por fase

### Após Treinamento (Test Set)
- **Ensemble:** 5 folds
- **TTA:** 4 transforms (original, flip_h, flip_v, rotate_90)
- **Threshold:** 0.5

### Métricas
- Test Dice (ensemble + TTA)
- Test IoU
- Per-class Dice (Exudates, Hemorrhages)
- Δ vs Baseline 0.6448

---

## 🎯 Meta de Sucesso

- **Mínimo aceitável:** Test Dice ≥ 0.6500 (+0.8%)
- **Bom resultado:** Test Dice ≥ 0.6600 (+2.4%)
- **Excelente:** Test Dice ≥ 0.6700 (+3.9%)

---

## 🧪 Alternativas a Testar (se falhar)

### Variação A: Gradual Unfreezing
- Fase 1: Só decoder (15 épocas)
- Fase 2: Decoder + block 6 (10 épocas)
- Fase 3: Decoder + blocks 5-6 (10 épocas)
- Fase 4: Tudo (5 épocas)

### Variação B: Longer Warm-up
- Fase 1: Só decoder (40 épocas, patience=15)
- Fase 2: Decoder + último bloco (20 épocas)
- Fase 3: Tudo (10 épocas)

### Variação C: Discriminative LR desde o início
```python
# Fase 1
optimizer = AdamW([
    {'params': decoder.parameters(), 'lr': 1e-3}
])

# Fase 2  
optimizer = AdamW([
    {'params': encoder.blocks[-1].parameters(), 'lr': 1e-4},
    {'params': decoder.parameters(), 'lr': 5e-4}
])

# Fase 3
optimizer = AdamW([
    {'params': encoder.blocks[0].parameters(), 'lr': 1e-5},
    {'params': encoder.blocks[1:-1].parameters(), 'lr': 5e-5},
    {'params': encoder.blocks[-1].parameters(), 'lr': 1e-4},
    {'params': decoder.parameters(), 'lr': 2e-4}
])
```

---

## 💡 Insights Esperados

**Se funcionar (Dice > 0.6500):**
- Freezing progressivo é efetivo para datasets pequenos
- Features do ImageNet são valiosas para ROP
- Treinamento em fases evita catastrófico forgetting

**Se falhar (Dice < 0.6448):**
- 54 imagens são poucas até para fine-tuning cuidadoso
- Baseline já estava otimizado
- Encoder pré-treinado pode não ser adequado para ROP

---

## 📂 Estrutura de Código

```python
# Pseudo-código do treinamento

def train_phase_1(model, train_loader, val_loader):
    """Treina apenas decoder"""
    # Congelar encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Otimizador
    optimizer = AdamW(model.decoder.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(...)
    early_stopping = EarlyStopping(patience=8)
    
    for epoch in range(25):
        train_loss, train_dice = train_epoch(...)
        val_loss, val_dice = validate_epoch(...)
        
        scheduler.step(val_dice)
        early_stopping(val_dice)
        
        if early_stopping.early_stop:
            break
    
    return model

def train_phase_2(model, train_loader, val_loader):
    """Fine-tuning parcial do encoder"""
    # Descongelar último bloco
    for param in model.encoder.blocks[-1].parameters():
        param.requires_grad = True
    
    # Otimizador com LRs diferentes
    optimizer = AdamW([
        {'params': model.encoder.blocks[-1].parameters(), 'lr': 1e-4},
        {'params': model.decoder.parameters(), 'lr': 5e-4}
    ])
    scheduler = ReduceLROnPlateau(...)
    early_stopping = EarlyStopping(patience=5)
    
    for epoch in range(15):
        train_loss, train_dice = train_epoch(...)
        val_loss, val_dice = validate_epoch(...)
        
        scheduler.step(val_dice)
        early_stopping(val_dice)
        
        if early_stopping.early_stop:
            break
    
    return model

def train_phase_3(model, train_loader, val_loader):
    """Refinamento global"""
    # Descongelar tudo
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = ReduceLROnPlateau(...)
    early_stopping = EarlyStopping(patience=5)
    
    for epoch in range(10):
        train_loss, train_dice = train_epoch(...)
        val_loss, val_dice = validate_epoch(...)
        
        scheduler.step(val_dice)
        early_stopping(val_dice)
        
        if early_stopping.early_stop:
            break
    
    return model

# Pipeline completo
for fold in range(5):
    model = EfficientNetUNet()
    
    # Fase 1
    print(f"Fold {fold} - Phase 1: Decoder Only")
    model = train_phase_1(model, train_loader, val_loader)
    
    # Fase 2
    print(f"Fold {fold} - Phase 2: Partial Fine-tuning")
    model = train_phase_2(model, train_loader, val_loader)
    
    # Fase 3
    print(f"Fold {fold} - Phase 3: Global Fine-tuning")
    model = train_phase_3(model, train_loader, val_loader)
    
    # Salvar best model
    torch.save(model.state_dict(), f'fold_{fold}_best.pth')

# Avaliar test set com ensemble + TTA
test_dice = evaluate_ensemble_tta(models, test_loader)
```

---

## ⚠️ Possíveis Problemas

1. **LRs diferentes requerem param groups:**
   - Solução: Usar dicionários no optimizer
   - Alternativa: LR única por fase (mais simples)

2. **Early stopping entre fases:**
   - Salvar checkpoint ao fim de cada fase
   - Carregar melhor checkpoint para próxima fase

3. **Scheduler restart entre fases:**
   - Criar novo scheduler a cada fase
   - ReduceLROnPlateau: resetar contador

4. **Tempo de treinamento:**
   - 3 fases × 5 folds = pode ser longo
   - Estimar: ~1-1.5h por fold × 5 = 5-7.5h total

---

## 📌 Checklist de Implementação

- [ ] Implementar função `freeze_encoder()`
- [ ] Implementar função `unfreeze_last_block()`
- [ ] Implementar função `unfreeze_all()`
- [ ] Criar optimizers com param groups
- [ ] Implementar ReduceLROnPlateau scheduler
- [ ] Adaptar early stopping para trabalhar em fases
- [ ] Salvar checkpoints entre fases
- [ ] Carregar splits de `outputs/cv_splits.json`
- [ ] Implementar ensemble + TTA
- [ ] Logging detalhado de cada fase
- [ ] Comparar com baseline 0.6448

---

**Estimativa de tempo:** 6-8 horas de GPU  
**Probabilidade de sucesso:** ⭐⭐⭐⭐ (4/5) - Técnica comprovada para datasets pequenos  
**Complexidade de implementação:** Média (precisa gerenciar fases)
