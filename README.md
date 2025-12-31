# ROP Segmentation Project

Projeto de segmentação de lesões de retinopatia diabética focado em **Exudatos** (Hard e Soft) e **Hemorragias**.

## Estrutura do Projeto

```
tapi_inrid/
├── configs/
│   ├── __init__.py
│   └── config.py              # Configurações e hiperparâmetros
├── data_factory/
│   ├── __init__.py
│   ├── data_factory.py        # Gerenciamento de metadados e splits
│   └── ROP_dataset.py         # PyTorch Dataset
├── utils/
│   ├── __init__.py
│   └── utils.py               # Funções auxiliares
├── notebooks/
│   └── data_exploration.ipynb # Análise exploratória
├── train_and_val_worker.py    # Treino e avaliação
├── main.py                    # Pipeline principal
├── requirements.txt           # Dependências
└── README.md
```

## Arquitetura

O projeto segue os princípios definidos no `AGENT.md`:

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

## Princípios de Design

Seguindo o `AGENT.md`:

✅ **Single Responsibility Principle**
- Cada classe tem uma responsabilidade clara
- Separação entre metadados e dados reais

✅ **Reprodutibilidade Científica**
- Seeds fixos
- GroupKFold para evitar leakage
- Configurações centralizadas

✅ **Clareza Arquitetural**
- Sem mistura de responsabilidades
- Código orientado à pesquisa
- Documentação clara

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

## Contribuindo

Este projeto segue diretrizes estritas de arquitetura definidas no `AGENT.md`. Ao contribuir:

1. Respeite a separação de responsabilidades
2. Não misture lógica de dados com treino
3. Mantenha configurações centralizadas
4. Documente mudanças

## Licença

Este projeto utiliza o dataset IDRiD, que possui licença CC-BY-4.0. Consulte `A. Segmentation/CC-BY-4.0.txt` para detalhes.

## Referências

- Dataset: [IDRiD - Indian Diabetic Retinopathy Image Dataset](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)
- Segmentation Models PyTorch: https://github.com/qubvel/segmentation_models.pytorch
