# Instruções para Agente - Projeto Segmentação ROP com Wavelets

## 📋 CONTEXTO DO PROJETO

### Visão Geral
Projeto de pesquisa de mestrado focado em **segmentação de lesões retinopatia** usando **U-Net + EfficientNet + Wavelet Skip Connections**.

**Dataset:** INRID (Indian Neonatal Retinopathy Database)
- 81 imagens totais (54 train/val + 27 test)
- Classes: Exsudatos e Hemorragias (multi-label)
- Resolução original: 4288×2848 pixels
- Resolução de treino: 512×512 pixels

**Melhor Modelo Atual:**
- Arquitetura: UNet + EfficientNet-B4 + Wavelet DWT (Haar) no primeiro skip
- Test Dice: **0.6721**
  - Exsudatos: 0.7275
  - Hemorragias: 0.6167
- Ganho vs baseline: +4.6%

---

## 🗂️ ESTRUTURA DO PROJETO (OBRIGATÓRIA)

```
tapi_inrid/
├── configs/                    # Configurações do projeto
│   ├── __init__.py
│   └── config.py              # Classe Config central
│
├── data_factory/              # Datasets e data loaders
│   ├── __init__.py
│   ├── data_factory.py        # DataFactory (cria datasets)
│   └── ROP_dataset.py         # Dataset customizado (ROPDataset)
│
├── docs/                      # 📝 DOCUMENTAÇÃO (sempre aqui!)
│   ├── ARQUITETURA_EXPLICADA.md
│   ├── TUTORIAL_WAVELETS.md
│   └── SUGESTOES_SLIDES.md
│
├── experiments/               # 🧪 SCRIPTS DE EXPERIMENTOS
│   ├── train_efficientnet_b1.py
│   ├── train_efficientnet_b2.py
│   ├── ...
│   ├── train_wavelet_skip1.py
│   └── exp_XXX_description.py  # NOVO formato (enumerar!)
│
├── logs/                      # Logs de treinamento
│   ├── training_efficientnet_b1.log
│   └── ...
│
├── models/                    # Arquiteturas de modelos
│   ├── __init__.py
│   ├── unet_efficientnet.py
│   └── unet_wavelet_skip1.py
│
├── notebooks/                 # Jupyter notebooks (exploração)
│   ├── data_exploration.ipynb
│   └── visualize_wavelet_predictions.ipynb
│
├── outputs/                   # Saídas de treinamento
│   ├── checkpoints/           # Modelos salvos (.pth)
│   │   ├── best_model_fold1.pth
│   │   └── ...
│   ├── cv_splits.json         # ⚠️ FROZEN! Não alterar
│   └── logs/                  # Logs detalhados (TensorBoard, etc.)
│
├── tests/                     # 🧪 TESTES (criar ANTES de experimentos!)
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_exp_XXX.py        # Teste do experimento XXX
│
├── utils/                     # Utilitários
│   ├── __init__.py
│   └── utils.py               # set_seed, métricas, etc.
│
├── main.py                    # Script principal (treinamento base)
├── train_and_val_worker.py    # Worker de treinamento
├── requirements.txt
└── README.md
```

---

## 📐 CONVENÇÕES DE NOMENCLATURA

### Experimentos (`experiments/`)

**FORMATO OBRIGATÓRIO (a partir de agora):**
```
exp_XXX_brief_description.py

Onde:
- XXX = número sequencial (001, 002, 003, ...)
- brief_description = descrição curta (snake_case)

Exemplos:
✅ exp_001_wavelet_skip2.py          # Testar wavelet no skip 2
✅ exp_002_attention_module.py        # Adicionar attention
✅ exp_003_multiscale_wavelet.py      # Wavelet multi-escala
✅ exp_004_clahe_rgb_only.py          # CLAHE apenas em R+B

❌ train_new_model.py                 # Não enumera
❌ test_wavelet.py                    # Confunde com testes
❌ experiment.py                      # Não descritivo
```

### Logs (`logs/`)
```
training_exp_XXX_brief_description.log

Exemplo:
✅ training_exp_001_wavelet_skip2.log
```

### Testes (`tests/`)
```
test_exp_XXX.py  # Testa o experimento XXX
test_<component>.py  # Testa componente específico

Exemplos:
✅ test_exp_001.py      # Testa exp_001_wavelet_skip2.py
✅ test_dataset.py      # Testa ROPDataset
✅ test_model.py        # Testa arquitetura
```

### Modelos (`models/`)
```
<architecture>_<variant>.py

Exemplos:
✅ unet_efficientnet.py          # U-Net base com EfficientNet
✅ unet_wavelet_skip1.py         # U-Net com wavelet no skip 1
✅ unet_attention.py             # U-Net com attention
```

### Documentação (`docs/`)
```
<TOPIC>_<TYPE>.md

Exemplos:
✅ ARQUITETURA_EXPLICADA.md
✅ TUTORIAL_WAVELETS.md
✅ EXPERIMENTOS_REALIZADOS.md
✅ INSTRUCOES_DEPLOYMENT.md
```

---

## 🔄 WORKFLOW DE EXPERIMENTOS (OBRIGATÓRIO)

### Passo 1: Criar Teste PRIMEIRO
```bash
# SEMPRE criar teste antes do experimento!

# Arquivo: tests/test_exp_XXX.py
"""
Teste do experimento XXX: <descrição>

Valida:
- Configuração carrega corretamente
- Dataset é criado sem erros
- Modelo inicializa
- Forward pass funciona
- Backward pass funciona (sem NaN, inf)
- Métricas são calculadas
"""

import pytest
import torch
from experiments.exp_XXX_description import (
    create_model,
    create_config,
    # outras funções
)

def test_config_creation():
    """Valida criação de config."""
    config = create_config()
    assert config is not None
    assert config.img_size == 512
    # ...

def test_model_initialization():
    """Valida inicialização do modelo."""
    model = create_model()
    assert model is not None
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params}")
    
def test_forward_pass():
    """Valida forward pass."""
    model = create_model()
    x = torch.randn(2, 3, 512, 512)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (2, 2, 512, 512)  # [B, C, H, W]
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_backward_pass():
    """Valida backward pass."""
    model = create_model()
    x = torch.randn(2, 3, 512, 512)
    target = torch.randint(0, 2, (2, 2, 512, 512)).float()
    
    output = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
    
    loss.backward()
    
    # Verificar gradientes
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf in {name}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Passo 2: Criar Experimento
```bash
# Arquivo: experiments/exp_XXX_description.py
"""
Experimento XXX: <Descrição Detalhada>

Objetivo:
- <objetivo principal>

Hipótese:
- <hipótese a ser testada>

Mudanças vs Baseline:
- <listar mudanças>

Baseline:
- <modelo de referência>

Resultados Esperados:
- <expectativa de ganho>
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from configs.config import Config
from data_factory.data_factory import DataFactory
from models.unet_new_variant import UNetNewVariant
from utils.utils import set_seed, DiceLoss
# ...

def create_config():
    """Cria configuração do experimento."""
    config = Config()
    
    # Modificações específicas
    config.experiment_name = "exp_XXX_description"
    config.num_epochs = 100
    config.learning_rate = 1e-4
    # ... outras configs
    
    return config

def create_model(config):
    """Cria modelo do experimento."""
    model = UNetNewVariant(
        encoder_name=config.encoder_name,
        encoder_weights='imagenet',
        in_channels=3,
        classes=config.num_classes
    )
    return model

def main():
    """Função principal de treinamento."""
    # Config
    config = create_config()
    set_seed(config.random_state)
    
    # Data
    data_factory = DataFactory(config)
    # ... criar dataloaders
    
    # Model
    model = create_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Training loop
    # ... implementar
    
    # Logging
    # ... salvar resultados

if __name__ == "__main__":
    main()
```

### Passo 3: Executar Teste
```bash
# SEMPRE rodar teste ANTES de treinar!
pytest tests/test_exp_XXX.py -v

# Se passar → prosseguir
# Se falhar → corrigir antes de treinar
```

### Passo 4: Executar Experimento
```bash
# Com logging
python experiments/exp_XXX_description.py 2>&1 | tee logs/training_exp_XXX_description.log

# Monitorar progresso
tail -f logs/training_exp_XXX_description.log
```

### Passo 5: Documentar Resultados
```bash
# Atualizar docs/EXPERIMENTOS_REALIZADOS.md

## Experimento XXX: <Descrição>

**Data:** YYYY-MM-DD

**Objetivo:**
- <objetivo>

**Configuração:**
```python
learning_rate = 1e-4
num_epochs = 100
# ...
```

**Resultados:**
| Métrica | Baseline | Exp XXX | Ganho |
|---------|----------|---------|-------|
| Test Dice | 0.6721 | 0.XXXX | +X.X% |
| Exsudatos | 0.7275 | 0.XXXX | +X.X% |
| Hemorragias | 0.6167 | 0.XXXX | +X.X% |

**Análise:**
- <conclusões>

**Status:** ✅ Sucesso / ❌ Falhou / ⚠️ Resultados inconclusivos
```

---

## ⚙️ USO DA CLASSE CONFIG (OBRIGATÓRIO)

### Regra Principal
**NUNCA criar variáveis globais para configurações em experimentos!**

A classe `Config` em `configs/config.py` já contém todos os hiperparâmetros padrão.
Cada experimento deve:
1. Criar um objeto `Config()`
2. Modificar APENAS os parâmetros que diferem do padrão
3. Usar `config.atributo` em todo o código

### Parâmetros Disponíveis na Config

```python
# configs/config.py - Parâmetros principais

@dataclass
class Config:
    # Dataset
    dataset_root: str = "/home/lucas/mestrado/tapi_inrid/A. Segmentation"
    classes: List[str] = ["exudates", "haemorrhages"]
    
    # Image preprocessing
    image_size: tuple = (512, 512)
    apply_clahe: bool = True  # NOTA: Setar False (já provou que piora)
    clahe_clip_limit: float = 2.0
    
    # Training
    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Model
    model_name: str = "unet"
    encoder_name: str = "efficientnet-b4"
    encoder_weights: str = "imagenet"
    num_classes: int = 2
    
    # Loss
    loss_type: str = "dice_focal"
    
    # Scheduler
    scheduler_type: str = "onecycle"  # ou "plateau"
    early_stopping_patience: int = 20
    
    # Cross-validation
    n_folds: int = 5
    random_state: int = 42
    
    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str  # auto-gerado em __post_init__
```

### Exemplo de Função create_config()

```python
def create_config():
    """
    Cria configuração do experimento.
    
    Usa a classe Config como BASE e modifica apenas o necessário.
    NUNCA criar variáveis globais separadas!
    """
    config = Config()
    
    # === Identificação ===
    config.experiment_name = "exp_XXX_description"
    
    # === Modificações específicas deste experimento ===
    config.apply_clahe = False  # Desabilitar CLAHE
    config.num_epochs = 100     # Mais épocas que padrão
    config.learning_rate = 1e-4 # LR diferente
    
    # === Para experimentos com novos parâmetros ===
    # Adicionar dinamicamente (dataclass permite)
    config.patch_size = 512
    config.patch_overlap = 50
    
    # === Checkpoint dir específico ===
    config.checkpoint_dir = os.path.join(config.output_dir, "checkpoints", "exp_XXX")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    return config
```

### ❌ ERRADO - Variáveis Globais

```python
# NÃO FAZER ISSO!
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
ENCODER_NAME = 'efficientnet-b4'

def main():
    model = smp.Unet(encoder_name=ENCODER_NAME, ...)  # ❌
    optimizer = Adam(lr=LEARNING_RATE)  # ❌
```

### ✅ CORRETO - Usar Config

```python
def create_config():
    config = Config()
    config.batch_size = 8
    config.num_epochs = 100
    config.learning_rate = 1e-4
    return config

def main():
    config = create_config()
    
    model = smp.Unet(encoder_name=config.encoder_name, ...)  # ✅
    optimizer = Adam(lr=config.learning_rate)  # ✅
    
    for epoch in range(config.num_epochs):  # ✅
        ...
```

### Benefícios

1. **Consistência**: Todos os experimentos usam a mesma estrutura
2. **Rastreabilidade**: Fácil salvar config em JSON para reprodutibilidade
3. **Valores padrão**: Não precisa redefinir tudo, só o que muda
4. **Evita bugs**: Parâmetros centralizados, não espalhados pelo código

---

## 🚨 ERROS COMUNS A EVITAR (LIÇÕES APRENDIDAS)

### 1. **PyTorch 2.6+ e Checkpoints**
```python
# ❌ ERRADO (causa erro em PyTorch 2.6+)
checkpoint = torch.load(path, map_location=device)

# ✅ CORRETO
checkpoint = torch.load(path, map_location=device, weights_only=False)

# ✅ MELHOR (se checkpoint tem dict)
checkpoint = torch.load(path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
```

**Razão:** PyTorch 2.6+ mudou default de `weights_only` para `True` por segurança.

### 2. **Estrutura de Checkpoint**
```python
# ❌ ERRADO (assume que checkpoint É o state_dict)
model.load_state_dict(checkpoint)

# ✅ CORRETO (checkpoint é dict com 'model_state_dict')
model.load_state_dict(checkpoint['model_state_dict'])

# ✅ SEMPRE salvar assim:
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'metrics': metrics
}, path)
```

### 3. **Multi-Label vs Multi-Class**
```python
# Nosso problema é MULTI-LABEL (não multi-class)!

# ❌ ERRADO (Softmax para multi-class)
output = torch.softmax(logits, dim=1)  # Classes mutuamente exclusivas

# ✅ CORRETO (Sigmoid para multi-label)
output = torch.sigmoid(logits)  # Classes independentes

# ❌ ERRADO (CrossEntropy para multi-class)
loss = nn.CrossEntropyLoss()

# ✅ CORRETO (BCEWithLogitsLoss para multi-label)
loss = nn.BCEWithLogitsLoss()
```

**Razão:** Pixel pode ter ambas classes (exsudato E hemorragia).

### 4. **CLAHE: Deve Aplicar em LAB, não RGB**
```python
# ❌ ERRADO (aplicar CLAHE diretamente em RGB)
clahe = cv2.createCLAHE(...)
for i in range(3):
    image[:,:,i] = clahe.apply(image[:,:,i])

# ✅ CORRETO (aplicar apenas em canal L do LAB)
lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)
l_clahe = clahe.apply(l)
lab_clahe = cv2.merge([l_clahe, a, b])
image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
```

**Lição:** CLAHE em todos os canais RGB causa oversegmentation (comprovado experimentalmente: 0.6428 com CLAHE vs 0.6501 sem CLAHE).

### 5. **GroupKFold e Data Leakage**
```python
# ❌ ERRADO (KFold normal - pode colocar imagens do mesmo paciente em train e val)
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)

# ✅ CORRETO (GroupKFold - garante pacientes separados)
from sklearn.model_selection import GroupKFold
kfold = GroupKFold(n_splits=5)

# Uso:
for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y, groups=patient_ids)):
    # train_idx e val_idx têm pacientes diferentes
```

**Razão:** Mesmo paciente tem imagens correlacionadas. GroupKFold evita vazamento.

### 6. **Cross-Validation Splits: FREEZE!**
```python
# ⚠️ IMPORTANTE: Salvar splits no primeiro experimento
import json

cv_splits = {
    'fold1': {'train': [...], 'val': [...]},
    # ...
}

with open('outputs/cv_splits.json', 'w') as f:
    json.dump(cv_splits, f)

# ✅ Todos os experimentos futuros DEVEM usar o mesmo split!
with open('outputs/cv_splits.json', 'r') as f:
    cv_splits = json.load(f)
```

**Razão:** Comparação justa entre experimentos.

### 7. **Normalização: ImageNet Stats**
```python
# ✅ SEMPRE usar mesma normalização do encoder pré-treinado
mean = [0.485, 0.456, 0.406]  # ImageNet mean
std = [0.229, 0.224, 0.225]   # ImageNet std

# Albumentations
A.Normalize(mean=mean, std=std)

# Desnormalizar para visualização
image_denorm = image * std + mean
image_denorm = np.clip(image_denorm, 0, 1)
```

### 8. **TTA (Test-Time Augmentation): Média, não Votação**
```python
# ❌ ERRADO (votação para segmentação)
predictions = [model(aug(x)) > 0.5 for aug in augs]
final = (sum(predictions) > len(predictions) // 2).float()

# ✅ CORRETO (média de probabilidades)
predictions = [torch.sigmoid(model(aug(x))) for aug in augs]
final = torch.stack(predictions).mean(dim=0)
final_binary = (final > 0.5).float()
```

### 9. **Dice Loss: Suavização é Crucial**
```python
# ❌ ERRADO (divisão por zero)
dice = (2 * intersection) / (pred.sum() + gt.sum())

# ✅ CORRETO (adicionar epsilon)
dice = (2 * intersection + 1e-8) / (pred.sum() + gt.sum() + 1e-8)

# ✅ MELHOR (epsilon maior para estabilidade)
dice = (2 * intersection + 1.0) / (pred.sum() + gt.sum() + 1.0)
```

### 10. **Masks: Valores 0 ou 255 → Normalizar para 0 ou 1**
```python
# ❌ ERRADO (assumir que mask já está em [0, 1])
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask_tensor = torch.from_numpy(mask)  # Valores 0 ou 255!

# ✅ CORRETO
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = (mask > 127).astype(np.float32)  # Binarizar para {0.0, 1.0}
mask_tensor = torch.from_numpy(mask)
```

### 11. **Encoding de Features: Congelar ou Não?**
```python
# Nosso caso: NÃO congelamos encoder

# ❌ Se fosse congelar (não fazemos):
for param in model.encoder.parameters():
    param.requires_grad = False

# ✅ Deixamos treinar end-to-end (fazemos):
# (nada a fazer, padrão é trainable)

# ⚠️ Usar encoder_weights='imagenet' (pré-treinado)
model = UNet(encoder_name='efficientnet-b4', encoder_weights='imagenet')
```

**Decisão:** Fine-tuning completo funcionou melhor que frozen encoder.

### 12. **Dependências Faltando (Hugging Face Hub)**
```python
# ❌ Erro comum ao carregar modelos pré-treinados:
# ModuleNotFoundError: No module named 'huggingface_hub'

# ✅ SEMPRE ter no requirements.txt:
torch>=2.0.0
torchvision>=0.15.0
segmentation-models-pytorch>=0.3.0
albumentations>=1.3.0
opencv-python>=4.7.0
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pyyaml>=6.0
huggingface-hub>=0.16.0  # ← IMPORTANTE!
ipywidgets>=8.0.0        # Para notebooks
```

---

## 🧬 CONTEXTO TÉCNICO COMPLETO

### Arquitetura Atual (Wavelet Skip 1)

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT: [B, 3, 512, 512]                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               ENCODER: EfficientNet-B4 (ImageNet)               │
│                                                                 │
│  features[0]: [B,   48, 512, 512]  ← Skip 0                    │
│  features[1]: [B,   48, 256, 256]  ← Skip 1 (WAVELET AQUI!) ✨│
│  features[2]: [B,   80, 128, 128]  ← Skip 2                    │
│  features[3]: [B,  192,  64,  64]  ← Skip 3                    │
│  features[4]: [B,  448,  32,  32]  ← Skip 4                    │
│  bottleneck:  [B, 1792,  16,  16]  ← Bottleneck                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    WAVELET TRANSFORM (Hook)                     │
│                                                                 │
│  features[1]: [B, 48, 256, 256]                                │
│       ↓                                                         │
│  DWT 2D (Haar):                                                 │
│       ↓                                                         │
│  LL (descartado): [B, 48, 128, 128]  ← Redundante com skip     │
│  LH (horizontal):  [B, 48, 128, 128]  ← Bordas horizontais      │
│  HL (vertical):    [B, 48, 128, 128]  ← Bordas verticais        │
│  HH (diagonal):    [B, 48, 128, 128]  ← Bordas diagonais        │
│       ↓                                                         │
│  concat(LH, HL, HH): [B, 144, 128, 128]                        │
│       ↓                                                         │
│  upsample → [B, 144, 256, 256]                                 │
│       ↓                                                         │
│  enhanced_skip = cat(features[1], wavelet)                     │
│                = [B, 192, 256, 256]                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DECODER: U-Net Upsampling                    │
│                                                                 │
│  Nível 5: [B, 1792, 16, 16] → [B, 448, 32, 32] + skip[4]      │
│  Nível 4: [B, 640,  32, 32] → [B, 192, 64, 64] + skip[3]      │
│  Nível 3: [B, 384,  64, 64] → [B,  80, 128,128] + skip[2]     │
│  Nível 2: [B, 160, 128,128] → [B,  48, 256,256] + skip[1]✨   │
│  Nível 1: [B, 240, 256,256] → [B,  48, 512,512] + skip[0]     │
│  Nível 0: [B,  96, 512,512] → [B,  32, 512,512]               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   OUTPUT HEAD: Conv 1×1                         │
│                                                                 │
│  [B, 32, 512, 512] → [B, 2, 512, 512]                          │
│                                                                 │
│  Channel 0: Exsudatos (logits)                                 │
│  Channel 1: Hemorragias (logits)                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    SIGMOID (Inference)                          │
│                                                                 │
│  [B, 2, 512, 512] → [B, 2, 512, 512] (probabilidades)          │
│                                                                 │
│  Threshold 0.5 → Máscaras binárias                             │
└─────────────────────────────────────────────────────────────────┘
```

**Parâmetros:**
- Total: 19,345,186 params
- Encoder: ~18M
- Decoder: ~1.3M
- Wavelet overhead: +9,312 (+0.05%)

### Hiperparâmetros Finais

```python
# Data
img_size = 512
apply_clahe = False  # Removido (piorava performance)
num_classes = 2

# Training
batch_size = 4
num_epochs = 100
learning_rate = 1e-4

# Optimizer
optimizer = AdamW(
    params=model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

# Loss
criterion = BCEWithLogitsLoss()

# Scheduler
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=7,
    verbose=True,
    min_lr=1e-7
)

# Early Stopping
patience = 15

# Data Augmentation (Train)
transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=15,
        p=0.5
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.3
    ),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

# Test-Time Augmentation
tta_transforms = [
    Original,
    HorizontalFlip,
    VerticalFlip,
    Rotate90,
    Rotate180,
    Rotate270
]

# Cross-Validation
cv_strategy = GroupKFold(n_splits=5)
cv_splits_path = 'outputs/cv_splits.json'  # FROZEN!
```

### Métricas

```python
def dice_score(pred, target, smooth=1.0):
    """
    Dice Score (F1 para segmentação).
    
    Args:
        pred: [B, C, H, W] (probabilidades ou binárias)
        target: [B, C, H, W] (ground truth)
        smooth: suavização (evita div por zero)
    
    Returns:
        dice: escalar [0, 1], 1 = perfeito
    """
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice.mean()  # Média sobre batch e classes

# Uso:
pred_binary = (torch.sigmoid(output) > 0.5).float()
dice = dice_score(pred_binary, target)
```

---

## 📊 ESTADO ATUAL (RESULTADOS)

### Progressão Completa

| Fase | Experimento | Test Dice | Ganho | Arquivo |
|------|-------------|-----------|-------|---------|
| 1 | B1 + CLAHE | 0.6272 | baseline | `train_efficientnet_b1.py` |
| 1 | B2 + CLAHE | 0.6265 | -0.1% | `train_efficientnet_b2.py` |
| 1 | B3 + CLAHE | 0.6257 | -0.2% | `train_efficientnet_b3.py` |
| 1 | **B4 + CLAHE** | **0.6428** | **+2.5%** | `train_efficientnet_b4_verify3.py` |
| 2 | B4 sem CLAHE | 0.6501 | +3.6% | `train_efficientnet_b4.py` |
| 3 | B4 Otimizado (100ep, 1e-4) | 0.6594 | +5.1% | `train_efficientnet_b4_verify3.py` |
| 4 | **Wavelet Skip 1** | **0.6721** | **+7.2%** | `train_wavelet_skip1.py` |

### Melhor Modelo (Wavelet Skip 1)

**Arquitetura:** `models/unet_wavelet_skip1.py`

**Resultados Detalhados:**
```
Cross-Validation (5-Fold):
- Fold 1: 0.6124
- Fold 2: 0.5789
- Fold 3: 0.6235
- Fold 4: 0.5918
- Fold 5: 0.5793
- Mean: 0.5972 ± 0.0292

Test Set (Ensemble + TTA):
- Overall Dice: 0.6721
- Exsudatos: 0.7275
- Hemorragias: 0.6167

Ganhos vs Baseline (B4 + CLAHE):
- Overall: +4.6%
- Exsudatos: +3.4%
- Hemorragias: +6.0% (maior benefício!)
```

**Checkpoints Salvos:**
```
outputs/checkpoints/
├── best_model_fold1.pth
├── best_model_fold2.pth
├── best_model_fold3.pth
├── best_model_fold4.pth
└── best_model_fold5.pth
```

---

## 🚀 PRÓXIMOS PASSOS SUGERIDOS

### Experimentos Prontos para Testar

#### 1. **exp_001_wavelet_skip2.py** (Alta Prioridade)
**Objetivo:** Testar wavelet no skip 2 (128×128)

**Hipótese:** Skip 2 pode capturar features de médio nível (entre detalhes finos e contexto global).

**Mudanças:**
```python
# Hook em features[2] ao invés de features[1]
# features[2]: [B, 80, 128, 128] → wavelet: [B, 240, 64, 64]
```

**Expectativa:** Dice ~0.6650-0.6700 (provavelmente pior que skip 1)

---

#### 2. **exp_002_multiscale_wavelet.py** (Média Prioridade)
**Objetivo:** Aplicar wavelet em MÚLTIPLOS skips (1 e 2)

**Hipótese:** Combinar bordas de alta e média resolução pode ser complementar.

**Mudanças:**
```python
# Hooks em features[1] E features[2]
# Skip 1: [B, 48, 256, 256] → wavelet: [B, 144, 256, 256]
# Skip 2: [B, 80, 128, 128] → wavelet: [B, 240, 128, 128]
```

**Expectativa:** Dice ~0.6750-0.6800 (+0.3-0.8%)

**Riscos:** Overhead de parâmetros (+~20k), overfitting

---

#### 3. **exp_003_attention_module.py** (Média Prioridade)
**Objetivo:** Adicionar Attention Gates no decoder

**Hipótese:** Attention pode focar em regiões de lesões, reduzindo falsos positivos.

**Mudanças:**
```python
# Adicionar Attention Gate antes de cada concatenação de skip
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        # ...
```

**Expectativa:** Dice ~0.6750-0.6800

**Riscos:** +5-10M parâmetros, treinamento mais lento

---

#### 4. **exp_004_daubechies_wavelet.py** (Baixa Prioridade)
**Objetivo:** Comparar Haar vs Daubechies (db2, db4)

**Hipótese:** Wavelets mais suaves podem capturar melhor bordas graduais.

**Mudanças:**
```python
# wavelet='db2' ao invés de 'haar'
wavelet_transform = WaveletTransform(wavelet='db2')
```

**Expectativa:** Dice ~0.6700-0.6730 (similar ou ligeiramente pior)

---

#### 5. **exp_005_dice_bce_combined_loss.py** (Média Prioridade)
**Objetivo:** Combinar Dice Loss + BCE Loss

**Hipótese:** Dice foca em overlap, BCE em pixel-wise accuracy. Combinação pode melhorar.

**Mudanças:**
```python
# Loss híbrida
loss = 0.5 * dice_loss(output, target) + 0.5 * bce_loss(output, target)
```

**Expectativa:** Dice ~0.6730-0.6770

---

#### 6. **exp_006_larger_image_size.py** (Alta Prioridade, mas Custoso)
**Objetivo:** Treinar com 768×768 ao invés de 512×512

**Hipótese:** Maior resolução preserva mais detalhes de lesões pequenas.

**Mudanças:**
```python
config.img_size = 768
config.batch_size = 2  # Reduzir por memória GPU
```

**Expectativa:** Dice ~0.6800-0.6900 (+1-2%)

**Riscos:** 
- Requer 2.25x memória GPU
- Treinamento 2x mais lento
- Pode precisar de +epochs

---

#### 7. **exp_007_test_time_scaling.py** (Baixa Prioridade)
**Objetivo:** Multi-scale testing (testar em 512, 768, 1024, averaging)

**Hipótese:** Diferentes escalas capturam diferentes níveis de detalhe.

**Mudanças:**
```python
# Inference em múltiplas escalas
scales = [512, 768, 1024]
predictions = []
for scale in scales:
    img_scaled = resize(img, scale)
    pred = model(img_scaled)
    pred_resized = resize(pred, 512)
    predictions.append(pred_resized)
    
final_pred = torch.stack(predictions).mean(dim=0)
```

**Expectativa:** Dice ~0.6750-0.6800

---

### Experimentos de Longo Prazo

#### 8. **exp_008_self_supervised_pretraining.py**
**Objetivo:** Pré-treinar encoder com task auto-supervisionada (e.g., rotation prediction, contrastive learning)

**Justificativa:** ImageNet é natural images, não médicas. Pré-treino específico pode ajudar.

---

#### 9. **exp_009_ensemble_heterogeneous.py**
**Objetivo:** Ensemble de arquiteturas diferentes (B4 + B3 + Wavelet)

**Justificativa:** Diversidade de modelos melhora ensemble.

---

#### 10. **exp_010_pseudo_labeling.py**
**Objetivo:** Usar modelo treinado para pseudo-rotular imagens não anotadas

**Justificativa:** Se houver imagens sem anotação, aproveitar.

---

## 📝 TEMPLATE PARA NOVOS DATASETS

Se criar dataset novo (e.g., para outro tipo de lesão), seguir estrutura:

```python
# Arquivo: data_factory/new_dataset.py

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from pathlib import Path

class NewDataset(Dataset):
    """
    Dataset para <descrição>.
    
    Args:
        dataframe: pd.DataFrame com colunas:
            - 'image_path': caminho da imagem
            - 'mask_<class>_paths': lista de caminhos de máscaras
            - 'patient_id': ID do paciente (para GroupKFold)
        config: objeto Config
        is_train: bool (True para augmentation)
        transform: transformação customizada (opcional)
    """
    
    def __init__(self, dataframe, config, is_train=True, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.config = config
        self.is_train = is_train
        
        # Criar transform se não fornecido
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
    
    def _get_default_transform(self):
        """Cria transformação padrão."""
        if self.is_train:
            return A.Compose([
                A.Resize(self.config.img_size, self.config.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                A.pytorch.ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.config.img_size, self.config.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                A.pytorch.ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict:
                'image': torch.Tensor [3, H, W]
                'mask': torch.Tensor [num_classes, H, W]
                'image_name': str
        """
        row = self.dataframe.iloc[idx]
        
        # Carregar imagem
        image = cv2.imread(row['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Carregar máscaras
        masks = []
        for class_name in self.config.classes:
            mask_paths = row[f'mask_{class_name}_paths']
            
            if isinstance(mask_paths, list) and len(mask_paths) > 0:
                # Combinar múltiplas máscaras (OR lógico)
                combined_mask = np.zeros(image.shape[:2], dtype=np.float32)
                for mask_path in mask_paths:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = (mask > 127).astype(np.float32)
                    combined_mask = np.maximum(combined_mask, mask)
                masks.append(combined_mask)
            else:
                # Sem máscara para essa classe
                masks.append(np.zeros(image.shape[:2], dtype=np.float32))
        
        masks = np.stack(masks, axis=-1)  # [H, W, num_classes]
        
        # Aplicar transformações
        transformed = self.transform(image=image, mask=masks)
        
        return {
            'image': transformed['image'],  # [3, H, W]
            'mask': transformed['mask'].permute(2, 0, 1),  # [num_classes, H, W]
            'image_name': Path(row['image_path']).stem
        }
```

---

## 🎯 PROTOCOLO DE COMUNICAÇÃO

### Como o Agente Deve Funcionar

1. **Sempre ler primeiro:**
   - `AGENTE_INSTRUCOES.md` (este arquivo)
   - `docs/EXPERIMENTOS_REALIZADOS.md` (histórico)
   - `outputs/cv_splits.json` (splits fixos)

2. **Antes de criar experimento:**
   - Verificar se experimento similar já foi feito
   - Verificar número sequencial (último exp_XXX)
   - Criar teste PRIMEIRO (`tests/test_exp_XXX.py`)

3. **Ao criar arquivos:**
   - Seguir estrutura de pastas obrigatória
   - Seguir convenções de nomenclatura
   - Adicionar docstrings detalhadas

4. **Durante treinamento:**
   - Sempre usar `2>&1 | tee logs/...` para logging
   - Monitorar com `tail -f`
   - Salvar checkpoints em `outputs/checkpoints/`

5. **Após experimento:**
   - Documentar em `docs/EXPERIMENTOS_REALIZADOS.md`
   - Atualizar tabelas de comparação
   - Se melhorar SOTA, destacar em bold

6. **Comunicação com usuário:**
   - Ser conciso (1-3 frases para confirmações simples)
   - Usar links markdown para arquivos: `[config.py](configs/config.py#L10)`
   - Evitar emojis (exceto quando usuário usa)
   - Explicar comandos técnicos antes de executar

7. **Tratamento de erros:**
   - Sempre verificar este arquivo (seção "ERROS COMUNS")
   - Se erro novo, documentar para próximos agentes
   - Não desistir facilmente, pesquisar soluções

---

## 📚 DOCUMENTAÇÃO OBRIGATÓRIA

### Manter Atualizados

1. **docs/EXPERIMENTOS_REALIZADOS.md**
   - Todos os experimentos (sucesso ou falha)
   - Tabela comparativa
   - Lições aprendidas

2. **docs/ARQUITETURA_EXPLICADA.md**
   - Se modificar arquitetura, atualizar diagrama
   - Explicar mudanças

3. **README.md** (raiz)
   - Setup instructions
   - Quick start
   - Resultados principais

4. **requirements.txt**
   - Se adicionar biblioteca, documentar versão

---

## 🔧 COMANDOS ÚTEIS

### Setup Inicial
```bash
# Criar ambiente
conda create -n rop_seg python=3.10
conda activate rop_seg

# Instalar dependências
pip install -r requirements.txt

# Verificar GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Testes
```bash
# Testar tudo
pytest tests/ -v

# Testar específico
pytest tests/test_exp_001.py -v

# Com coverage
pytest tests/ --cov=. --cov-report=html
```

### Treinamento
```bash
# Experimento com logging
python experiments/exp_XXX_description.py 2>&1 | tee logs/training_exp_XXX.log

# Monitorar progresso
tail -f logs/training_exp_XXX.log

# Em background (tmux)
tmux new -s exp_XXX
python experiments/exp_XXX.py 2>&1 | tee logs/training_exp_XXX.log
# Ctrl+B D para detach
```

### Análise de Resultados
```bash
# Extrair métricas finais
tail -50 logs/training_exp_XXX.log | grep -E "(Test Results|Dice|Exudates|Haemorrhages)"

# Comparar com baseline
grep "Test Dice" logs/training_*.log

# Ver CV folds
python -c "import json; print(json.load(open('outputs/cv_splits.json')))"
```

### Limpeza
```bash
# Limpar checkpoints antigos (CUIDADO!)
# (Manter apenas best de cada fold)
find outputs/checkpoints/ -name "epoch_*.pth" -delete

# Limpar cache
rm -rf __pycache__ */__pycache__
```

---

## 🎓 RECURSOS DE APRENDIZADO

### Papers Implementados
1. **U-Net:** Ronneberger et al., 2015
2. **EfficientNet:** Tan & Le, 2019
3. **Wavelets em CNNs:** Liu et al., 2019
4. **Attention U-Net:** Oktay et al., 2018

### Tutoriais Criados
- `docs/TUTORIAL_WAVELETS.md` - Wavelet completo
- `docs/ARQUITETURA_EXPLICADA.md` - Arquitetura atual
- `docs/SUGESTOES_SLIDES.md` - Material para apresentação

---

## ✅ CHECKLIST PARA NOVOS AGENTES

Antes de começar qualquer tarefa:

- [ ] Li `AGENTE_INSTRUCOES.md` completo
- [ ] Li `docs/EXPERIMENTOS_REALIZADOS.md` (histórico)
- [ ] Verifiquei estrutura de pastas
- [ ] Verifiquei último número de experimento
- [ ] Entendi estado atual (melhor modelo, resultados)
- [ ] Sei onde salvar cada tipo de arquivo
- [ ] Sei como nomear arquivos
- [ ] Sei que preciso criar teste ANTES de experimento
- [ ] Sei como usar `outputs/cv_splits.json`
- [ ] Li seção "ERROS COMUNS"

---

## 🆘 TROUBLESHOOTING RÁPIDO

| Problema | Solução |
|----------|---------|
| `ModuleNotFoundError: huggingface_hub` | `pip install huggingface-hub` |
| `RuntimeError: expected scalar type Float but found Byte` | Normalizar mask para [0, 1]: `mask = (mask > 127).astype(np.float32)` |
| `CUDA out of memory` | Reduzir batch_size em config.py |
| `NotImplementedError: weights_only` | Adicionar `weights_only=False` em `torch.load()` |
| `KeyError: 'model_state_dict'` | Checkpoint pode ser só state_dict: `model.load_state_dict(checkpoint)` |
| `NaN loss during training` | Verificar normalização, learning rate, gradient clipping |
| Test Dice muito baixo (<0.5) | Verificar se usa Sigmoid (não Softmax), se mask está [0,1] |
| CV splits diferentes entre runs | Usar `outputs/cv_splits.json` fixo, não gerar novo |

---

## 🚦 QUANDO PEDIR AJUDA AO USUÁRIO

**SEMPRE perguntar se:**
- Experimento novo não tem precedente claro
- Mudança pode afetar comparação com resultados anteriores
- Overhead de compute é muito alto (>2x tempo de treino)
- Decisão arquitetural tem trade-offs não óbvios

**NUNCA perguntar se:**
- Solução está documentada em "ERROS COMUNS"
- É tarefa rotineira (criar teste, rodar experimento)
- Informação está disponível nos arquivos do projeto

---

## 📞 CONTATO E CONTEXTO

**Projeto:** Mestrado em Segmentação de ROP  
**Dataset:** INRID (81 imagens, 2 classes)  
**Status:** Fase de otimização (baseline estabelecido)  
**Melhor Modelo:** Wavelet Skip 1 (0.6721 Test Dice)  

**Objetivo Final:** Publicar artigo com modelo state-of-the-art para segmentação de lesões em ROP.

---

**Última Atualização:** 2026-01-10  
**Versão:** 1.0  
**Autor:** Lucas (com assistência de GitHub Copilot)

---

## 🎯 RESUMO EXECUTIVO (TL;DR)

```
ESTRUTURA: configs/ data_factory/ docs/ experiments/ logs/ models/ notebooks/ outputs/ tests/ utils/

NOMENCLATURA: 
- Experimentos: exp_XXX_description.py
- Logs: training_exp_XXX_description.log
- Testes: test_exp_XXX.py (criar PRIMEIRO!)

WORKFLOW:
1. Criar teste → 2. Criar experimento → 3. Rodar teste → 4. Treinar → 5. Documentar

ERROS COMUNS:
- PyTorch 2.6: weights_only=False
- Multi-label: Sigmoid + BCEWithLogitsLoss (não Softmax + CrossEntropy)
- CLAHE: Aplicar em LAB-L, não RGB
- GroupKFold: Por paciente
- CV Splits: outputs/cv_splits.json (FROZEN!)

ESTADO ATUAL:
- Melhor: Wavelet Skip 1 (0.6721)
- Baseline: B4 + CLAHE (0.6428)
- Ganho: +4.6%

PRÓXIMOS PASSOS:
- exp_001: Wavelet Skip 2
- exp_002: Multi-scale Wavelet
- exp_003: Attention Gates
- exp_006: Larger Image Size (768×768)
```

---

**FIM DAS INSTRUÇÕES**

Boa sorte, novo agente! 🚀 Você tem tudo que precisa para continuar este projeto de excelência.
