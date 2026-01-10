#!/bin/bash

# Script para executar o experimento de Patch-Based Segmentation
# Uso: bash run_patch_experiment.sh

set -e  # Exit on error

echo "========================================================================"
echo "  Experimento 12: Patch-Based Segmentation"
echo "========================================================================"
echo ""

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar se estamos no diretório correto
if [ ! -f "experiments/train_patch_based.py" ]; then
    echo -e "${RED}Erro: Execute este script do diretório raiz do projeto${NC}"
    exit 1
fi

echo -e "${YELLOW}[1/3] Testando dataset de patches...${NC}"
python tests/test_patch_dataset.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Erro nos testes do dataset. Abortando.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Testes passaram!${NC}"
echo ""
echo -e "${YELLOW}[2/3] Iniciando treinamento...${NC}"
echo ""
echo "Configuração:"
echo "  - Patch size: 512×512"
echo "  - Overlap: 50px"
echo "  - Total patches treino: ~3,780"
echo "  - Folds: 5"
echo "  - Epochs: 50"
echo ""

read -p "Deseja continuar com o treinamento? (s/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Ss]$ ]]; then
    echo "Treinamento cancelado."
    exit 0
fi

echo ""
echo "Iniciando treinamento... (isso pode levar várias horas)"
echo ""

# Executar treinamento
python experiments/train_patch_based.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Erro durante o treinamento.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Treinamento completo!${NC}"
echo ""
echo -e "${YELLOW}[3/3] Verificando resultados...${NC}"

# Verificar se resultados foram salvos
if [ -f "outputs/patch_based_results.json" ]; then
    echo -e "${GREEN}✓ Resultados salvos em outputs/patch_based_results.json${NC}"
    echo ""
    echo "Resumo dos resultados:"
    python -c "
import json
with open('outputs/patch_based_results.json', 'r') as f:
    results = json.load(f)
print(f\"  CV Mean Dice: {results['mean_cv_dice']:.4f} ± {results['std_cv_dice']:.4f}\")
print(f\"  Test Dice:    {results['test_results']['mean_dice']:.4f}\")
print(f\"  Test IoU:     {results['test_results']['mean_iou']:.4f}\")
print()
print('Per-class (Test):')
for class_name in ['exudates', 'haemorrhages']:
    dice = results['test_results'][f'{class_name}_dice']
    iou = results['test_results'][f'{class_name}_iou']
    print(f\"  {class_name:12} - Dice: {dice:.4f}, IoU: {iou:.4f}\")
"
else
    echo -e "${RED}Aviso: Arquivo de resultados não encontrado${NC}"
fi

echo ""
echo "========================================================================"
echo -e "${GREEN}  Experimento Completo!${NC}"
echo "========================================================================"
echo ""
echo "Arquivos gerados:"
echo "  - outputs/patch_based_results.json"
echo "  - outputs/checkpoints/patch_based/*.pth"
echo "  - outputs/patch_visualization.png"
echo ""
echo "Próximos passos:"
echo "  1. Analisar resultados em outputs/patch_based_results.json"
echo "  2. Comparar com baseline (0.6448 Test Dice)"
echo "  3. Visualizar predições"
echo ""
