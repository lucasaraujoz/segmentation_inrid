# ROP Segmentation - TAPI INRID

Projeto de segmentação de lesões em imagens de retina para detecção de Retinopatia da Prematuridade (ROP).

## 🎯 Melhor Resultado

**Test Dice: 0.6448** (Ensemble 5-fold + TTA)

- **Arquitetura:** EfficientNet-B4 + UNet
- **Dataset:** 54 treino / 27 teste
- **Pré-processamento:** CLAHE LAB L-channel
- **Ensemble:** 5 folds + 4 transformações TTA

## 📁 Estrutura do Projeto

```
tapi_inrid/
├── 📂 configs/              # Configurações do projeto
├── 📂 data_factory/         # Dataset loaders e transforms
├── 📂 models/               # Arquiteturas de modelos
├── 📂 utils/                # Funções utilitárias
│
├── 📂 experiments/          # Scripts de treinamento dos experimentos
│   ├── README.md            # Guia dos experimentos
│   ├── verify_baseline.py   # Reproduzir baseline (0.6448) !
│   └── train_*.py           # Outros experimentos
│
├── 📂 docs/                 # Documentação
│   ├── README.md            # Guia da documentação
│   └── EXPERIMENTOS.md      # Análise completa experimentos
│
├── 📂 logs/                 # Logs de todos os treinamentos
│   └── README.md            # Guia dos logs
│
├── 📂 outputs/              # Resultados e checkpoints
│   ├── checkpoints/         # Modelos treinados
│   │   └── baseline_verify/ # Melhor modelo (0.6448)
│   └── *.json               # Resultados em JSON
│
├── 📂 notebooks/            # Jupyter notebooks para análise
├── 📂 A. Segmentation/      # Dataset original
│
├── main.py                  # Script principal de treinamento
└── requirements.txt         # Dependências Python
```

## 🚀 Quick Start

### 1. Instalação
```bash
pip install -r requirements.txt
```

### 2. Reproduzir Melhor Resultado
```bash
python experiments/verify_baseline.py
```

### 3. Avaliar Test Set
```bash
python experiments/evaluate_test_ensemble.py
```

## 📊 Experimentos Realizados

Total: **10 experimentos completos** + 3 interrompidos

### Ranking de Resultados

| # | Experimento | Test Dice | Δ vs Baseline | Status |
|---|-------------|-----------|---------------|--------|
| 1 | **Baseline (EfficientNet-B4 + UNet)** | **0.6448** | **0.00%** | ✅ **MELHOR** |
| 2 | Extreme Augmentation | 0.6422 | -0.40% | ❌ |
| 3 | ASPP Bottleneck | 0.6230 | -3.30% | ❌ |
| 4 | Attention Gates (Fixed) | 0.6182 | -4.13% | ❌ |
| 5 | Moderate Augmentation | 0.6009 | -6.80% | ❌ |
| 6 | ASPP Decoder | 0.5947 | -7.77% | ❌ |
| 7 | Green Channel CLAHE | CV: 0.5212 | -5.59% | ❌ Interrompido |
| 8 | Attention Gates (Buggy) | 0.5109 | -20.69% | ❌ Bug |
| 9 | Boundary Loss | 0.0100 | -99.0% | ❌ Falha |
| - | Frangi Enhancement | N/A | N/A | ❌ Abandonado |

**Ver análise completa:** [docs/EXPERIMENTOS.md](docs/EXPERIMENTOS.md)

## 🔍 Principal Descoberta

**Dataset muito pequeno (54 imagens) limita melhorias:**

❌ Arquiteturas complexas → Overfitting  
❌ Augmentação avançada → Piora resultados  
❌ Processamento de imagem → Perde informação  
✅ **Baseline simples é o melhor para este dataset**

## 📖 Documentação

- **[docs/EXPERIMENTOS.md](docs/EXPERIMENTOS.md)** - Análise detalhada de todos os experimentos
  - Configurações completas
  - Resultados e métricas
  - Análises técnicas profundas
  - Insights e lições aprendidas
  
- **[experiments/README.md](experiments/README.md)** - Guia dos scripts de treinamento

- **[logs/README.md](logs/README.md)** - Guia dos logs de treinamento

## Arquitetura do Baseline

O melhor resultado usa os princípios:

### 1. **Config** (`configs/config.py`)
- Gerencia todos os hiperparâmetros
- Define paths do dataset
- Configurações de pré-processamento (CLAHE)
- Parâmetros de treino

### 2. **DataFactory** (`data_factory/data_factory.py`)
- Mapeia estrutura de diretórios do dataset
- Cria DataFrame com metadados (paths de imagens e máscaras)
- Prepara splits para cross-validation com GroupKFold
- **Nunca carrega imagens, apenas metadados**

### 3. **ROPDataset** (`data_factory/ROP_dataset.py`)
- PyTorch Dataset
- Carrega imagens (.jpg) e máscaras (.tiff)
- Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Combina múltiplas máscaras (Hard + Soft Exudates)
- Suporta data augmentation com Albumentations

### 4. **TrainAndEvalWorker** (`train_and_val_worker.py`)
- Gerencia treinamento e validação
- Cria DataLoaders
- Instancia modelos (U-Net, U-Net++, DeepLabV3+)
- Calcula métricas (Dice, IoU)
- Salva checkpoints

### 5. **Main** (`main.py`)
- Orquestra o pipeline completo
- Apenas instancia classes e chama métodos
- **Sem lógica de treino ou carregamento de dados**

## Dataset

O projeto utiliza o dataset **IDRiD** (Indian Diabetic Retinopathy Image Dataset):

```
A. Segmentation/
├── 1. Original Images/
│   ├── a. Training Set/
│   └── b. Testing Set/
└── 2. All Segmentation Groundtruths/
    ├── a. Training Set/
    │   ├── 2. Haemorrhages/
    │   ├── 3. Hard Exudates/
    │   └── 4. Soft Exudates/
    └── b. Testing Set/
        ├── 2. Haemorrhages/
        ├── 3. Hard Exudates/
        └── 4. Soft Exudates/
```

### Classes Consideradas
- **Exudatos**: Hard e Soft Exudates combinados em uma única classe
- **Hemorragias**: Hemorrhages

**Nota**: Microaneurismas e Optic Disc são ignorados.

## Instalação

```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

## Uso

### 1. Exploração de Dados

Abra o notebook de análise exploratória:

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

O notebook mostra:
- Distribuição de classes
- Exemplos de imagens e máscaras
- Efeito do pré-processamento CLAHE
- Estatísticas do dataset

### 2. Treinamento

Execute o pipeline principal:

```bash
python main.py
```

O script irá:
1. Criar metadados do dataset
2. Preparar splits para 5-fold cross-validation
3. Treinar modelo U-Net com encoder ResNet34
4. Avaliar no conjunto de teste

### 3. Configuração

Edite [configs/config.py](configs/config.py) para ajustar:
- Tamanho das imagens
- Hiperparâmetros de treino
- Arquitetura do modelo
- Configurações de CLAHE

## Características Técnicas

### Pré-processamento
- **CLAHE**: Aplicado no canal L (luminosidade) do espaço LAB
- **Normalização**: ImageNet mean/std
- **Resize**: 512x512 (configurável)

### Data Augmentation
- Horizontal/Vertical flip
- Random rotation (90°)
- ShiftScaleRotate
- Elastic/Grid/Optical distortion

### Modelo
- **Arquitetura**: U-Net (padrão)
- **Encoder**: ResNet34 pré-treinado (ImageNet)
- **Loss**: Binary Cross Entropy with Logits
- **Métricas**: Dice Score, IoU

### Cross-Validation
- **GroupKFold** (5 folds)
- Agrupamento por paciente para evitar data leakage

## Estrutura de Saída

```
outputs/
├── checkpoints/
│   ├── best_model_fold1.pth
│   ├── best_model_fold2.pth
│   └── ...
└── logs/
    └── (logs de treinamento)
```

## Métricas

- **Dice Score**: Métrica principal
- **IoU (Jaccard Index)**: Métrica complementar
- Calculadas por classe e média geral

