"""
Experimento 013: Patch-Based Training com Controle Rigoroso de Leakage

================================================================================
OBJETIVO:
- Treinar modelo usando resolução original das imagens (4288×2848)
- Dividir em patches de 512×512 para preservar detalhes finos
- GARANTIR que patches de uma mesma imagem/paciente NUNCA apareçam em splits diferentes

================================================================================
HIPÓTESE:
- Resolução original preserva características de lesões pequenas perdidas no resize
- Mais patches = mais dados de treinamento (70 patches/imagem vs 1 imagem)
- Overlap entre patches melhora reconstrução durante inferência

================================================================================
CONTROLE DE LEAKAGE (FUNDAMENTAL):
- Split é feito NO NÍVEL DE IMAGENS, não de patches
- Primeiro: dividir imagens em train/val (via frozen_cv_splits.json)
- Depois: extrair patches apenas das imagens de cada split
- NUNCA: patch da imagem X em train e outro patch da imagem X em val

Fluxo:
1. Carregar frozen_cv_splits.json (train_indices, val_indices por fold)
2. Para cada fold:
   a. train_images = df.iloc[train_indices]  
   b. val_images = df.iloc[val_indices]
   c. train_patches = extrair_patches(train_images)
   d. val_patches = extrair_patches(val_images)
3. Treinar com train_patches, validar com val_patches
4. Na inferência: reconstruir imagem completa a partir dos patches

================================================================================
BASELINE:
- UNet + EfficientNet-B4 com resize para 512×512: Dice = 0.6721

RESULTADOS ESPERADOS:
- Melhoria em lesões pequenas (exsudatos pontuais, micro-hemorragias)
- Potencial aumento de 2-5% no Dice por preservar detalhes

================================================================================
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

from configs.config import Config
from data_factory.data_factory import DataFactory
from data_factory.ROP_dataset_patches import ROPDatasetPatches
from utils.utils import set_seed, calculate_dice_score, calculate_iou_score


# ==============================================================================
# CONFIGURAÇÃO DO EXPERIMENTO
# ==============================================================================

def create_config():
    """
    Cria configuração do experimento.
    
    Usa a classe Config como base e modifica apenas os parâmetros necessários.
    Isso garante consistência com outros experimentos e evita duplicação.
    """
    config = Config()
    
    # === Identificação do experimento ===
    config.experiment_name = "exp_013_patch_leakage_free"
    
    # === Configurações de Patch ===
    # Novos atributos específicos para patch-based training
    config.patch_size = 512
    config.patch_overlap = 50  # 10% overlap para reconstrução suave
    config.patch_stride = config.patch_size - config.patch_overlap  # 462
    
    # === Preprocessing ===
    config.apply_clahe = False  # IMPORTANTE: Sem CLAHE (já provou que piora)
    
    # === Training ===
    config.batch_size = 8  # Pode ser maior que baseline
    config.num_epochs = 100
    config.learning_rate = 1e-4
    config.weight_decay = 1e-4
    
    # === Model ===
    config.encoder_name = 'efficientnet-b4'
    config.encoder_weights = 'imagenet'
    
    # === Scheduler (override para ReduceLROnPlateau) ===
    config.scheduler_type = 'plateau'
    config.early_stopping_patience = 15
    
    # === Checkpoint dir específico ===
    config.checkpoint_dir = os.path.join(config.output_dir, "checkpoints", "patch_based_v2")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    return config


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def load_frozen_splits(config):
    """Carrega os splits frozen para garantir comparação justa."""
    frozen_path = os.path.join(config.output_dir, 'frozen_cv_splits.json')
    
    if not os.path.exists(frozen_path):
        # Fallback para cv_splits.json
        frozen_path = os.path.join(config.output_dir, 'cv_splits.json')
        print(f"⚠️ Usando cv_splits.json (frozen_cv_splits.json não encontrado)")
    
    with open(frozen_path, 'r') as f:
        splits = json.load(f)
    
    print(f"✓ Splits carregados de: {frozen_path}")
    print(f"  - {len(splits['folds'])} folds")
    print(f"  - {splits['metadata']['train_samples']} imagens de treino")
    print(f"  - {splits['metadata']['test_samples']} imagens de teste")
    
    return splits


def verify_no_leakage(train_df, val_df, fold_num):
    """Verifica que não há vazamento entre train e val."""
    train_images = set(train_df['image_name'].tolist())
    val_images = set(val_df['image_name'].tolist())
    
    overlap = train_images & val_images
    
    if len(overlap) > 0:
        raise ValueError(
            f"❌ LEAKAGE DETECTADO no Fold {fold_num}!\n"
            f"Imagens em train E val: {overlap}"
        )
    
    print(f"  ✓ Fold {fold_num}: Sem leakage ({len(train_images)} train, {len(val_images)} val)")


def reconstruct_from_patches(patches_pred, patch_info, img_width, img_height, 
                             patch_size=512, num_classes=2):
    """
    Reconstrói predição da imagem completa a partir de patches.
    
    Regiões de overlap são calculadas pela MÉDIA das predições.
    
    Args:
        patches_pred: np.array [N_patches, num_classes, H, W]
        patch_info: lista de dicts com 'patch_x', 'patch_y'
        img_width: largura original
        img_height: altura original
        patch_size: tamanho do patch
        num_classes: número de classes
        
    Returns:
        np.array [num_classes, img_height, img_width]
    """
    # Acumuladores
    full_pred = np.zeros((num_classes, img_height, img_width), dtype=np.float32)
    counts = np.zeros((img_height, img_width), dtype=np.float32)
    
    # Acumular predições
    for i, info in enumerate(patch_info):
        x, y = info['patch_x'], info['patch_y']
        patch = patches_pred[i]  # [C, H, W]
        
        full_pred[:, y:y+patch_size, x:x+patch_size] += patch
        counts[y:y+patch_size, x:x+patch_size] += 1
    
    # Média nas regiões de overlap
    counts = np.maximum(counts, 1)  # Evitar divisão por zero
    full_pred = full_pred / counts[np.newaxis, :, :]
    
    return full_pred


def calculate_patches_stats(dataframe, patch_size, overlap):
    """Calcula estatísticas de patches para um DataFrame."""
    # Assumindo imagens de 4288x2848 (INRID padrão)
    img_w, img_h = 4288, 2848
    stride = patch_size - overlap
    
    n_w = int(np.ceil((img_w - patch_size) / stride)) + 1
    n_h = int(np.ceil((img_h - patch_size) / stride)) + 1
    patches_per_img = n_w * n_h
    
    total_patches = len(dataframe) * patches_per_img
    
    return {
        'images': len(dataframe),
        'patches_per_image': patches_per_img,
        'total_patches': total_patches,
        'grid_size': f"{n_w}×{n_h}"
    }


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

class PatchTrainer:
    """Trainer para patch-based segmentation."""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # Criar modelo
        self.model = smp.Unet(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=3,
            classes=config.num_classes
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=7,
            min_lr=1e-7
        )
        
        # Loss
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Tracking
        self.best_val_dice = 0
        self.patience_counter = 0
        self.patience = config.early_stopping_patience
    
    def train_epoch(self, train_loader):
        """Treina uma época."""
        self.model.train()
        
        total_loss = 0
        total_dice = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Loss
            loss = self.criterion(outputs, masks)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                preds = torch.sigmoid(outputs) > 0.5
                dice = calculate_dice_score(preds.float(), masks).mean()
            
            total_loss += loss.item()
            total_dice += dice.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{dice.item():.4f}"
            })
        
        return {
            'loss': total_loss / num_batches,
            'dice': total_dice / num_batches
        }
    
    def validate_epoch(self, val_loader):
        """Valida uma época."""
        self.model.eval()
        
        total_loss = 0
        total_dice = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                preds = torch.sigmoid(outputs) > 0.5
                dice = calculate_dice_score(preds.float(), masks).mean()
                
                total_loss += loss.item()
                total_dice += dice.item()
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'dice': total_dice / num_batches
        }
    
    def train_fold(self, train_loader, val_loader, fold, checkpoint_dir):
        """Treina um fold completo."""
        best_model_path = os.path.join(checkpoint_dir, f"best_model_fold{fold+1}_patch.pth")
        
        for epoch in range(NUM_EPOCHS):
            print(f"\n  Epoch {epoch+1}/{NUM_EPOCHS}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Scheduler step
            self.scheduler.step(val_metrics['dice'])
            
            # Print metrics
            lr = self.optimizer.param_groups[0]['lr']
            print(f"    Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}")
            print(f"    Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, LR: {lr:.2e}")
            
            # Check best
            if val_metrics['dice'] > self.best_val_dice:
                self.best_val_dice = val_metrics['dice']
                self.patience_counter = 0
                
                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_dice': val_metrics['dice'],
                    'val_loss': val_metrics['loss']
                }, best_model_path)
                
                print(f"    ★ New best! Dice: {self.best_val_dice:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"    Early stopping after {epoch+1} epochs")
                    break
        
        return {
            'best_val_dice': self.best_val_dice,
            'best_model_path': best_model_path
        }
    
    def reset_for_fold(self):
        """Reseta trainer para novo fold."""
        config = self.config
        
        # Re-criar modelo
        self.model = smp.Unet(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=3,
            classes=config.num_classes
        ).to(self.device)
        
        # Re-criar optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Re-criar scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=7,
            min_lr=1e-7
        )
        
        # Reset tracking
        self.best_val_dice = 0
        self.patience_counter = 0


def evaluate_on_test(trainer, test_df, config, checkpoint_paths):
    """
    Avalia no test set usando ensemble dos melhores modelos.
    
    Reconstrói imagens completas a partir de patches.
    """
    print("\n" + "="*80)
    print("Avaliação no Test Set (com reconstrução de patches)")
    print("="*80)
    
    device = trainer.device
    
    # Criar dataset de teste com patches
    test_dataset = ROPDatasetPatches(
        dataframe=test_df,
        config=config,
        patch_size=config.patch_size,
        overlap=config.patch_overlap,
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"\nTest: {len(test_df)} imagens → {len(test_dataset)} patches")
    
    # Coletar predições de todos os patches
    all_preds = []
    all_masks = []
    all_info = []
    
    # Usar ensemble (média dos modelos de cada fold)
    models = []
    for path in checkpoint_paths:
        model = smp.Unet(
            encoder_name=config.encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=config.num_classes
        ).to(device)
        
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
    
    print(f"Usando ensemble de {len(models)} modelos")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting")):
            images = batch['image'].to(device)
            masks = batch['mask'].cpu().numpy()
            
            # Ensemble prediction
            preds_ensemble = []
            for model in models:
                outputs = model(images)
                preds = torch.sigmoid(outputs).cpu().numpy()
                preds_ensemble.append(preds)
            
            # Média do ensemble
            preds_avg = np.mean(preds_ensemble, axis=0)
            
            all_preds.append(preds_avg)
            all_masks.append(masks)
            
            # Salvar info de cada patch
            batch_size = len(batch['image_idx'])
            for i in range(batch_size):
                all_info.append({
                    'image_idx': batch['image_idx'][i].item(),
                    'image_name': batch['image_name'][i],
                    'patch_x': batch['patch_x'][i].item(),
                    'patch_y': batch['patch_y'][i].item(),
                    'img_width': batch['img_width'][i].item(),
                    'img_height': batch['img_height'][i].item()
                })
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    print(f"Total patches processados: {len(all_preds)}")
    
    # Agrupar patches por imagem
    from collections import defaultdict
    image_patches = defaultdict(lambda: {'preds': [], 'masks': [], 'info': []})
    
    for idx, info in enumerate(all_info):
        img_name = info['image_name']
        image_patches[img_name]['preds'].append(all_preds[idx])
        image_patches[img_name]['masks'].append(all_masks[idx])
        image_patches[img_name]['info'].append({
            'patch_x': info['patch_x'],
            'patch_y': info['patch_y']
        })
        image_patches[img_name]['img_width'] = info['img_width']
        image_patches[img_name]['img_height'] = info['img_height']
    
    # Reconstruir e calcular métricas por imagem
    print("\nReconstruindo imagens e calculando métricas...")
    
    all_dice_exudates = []
    all_dice_haemorrhages = []
    
    for img_name, data in image_patches.items():
        preds = np.array(data['preds'])
        masks = np.array(data['masks'])
        
        # Reconstruir imagem completa
        full_pred = reconstruct_from_patches(
            preds, data['info'],
            data['img_width'], data['img_height'],
            config.patch_size, config.num_classes
        )
        
        full_mask = reconstruct_from_patches(
            masks, data['info'],
            data['img_width'], data['img_height'],
            config.patch_size, config.num_classes
        )
        
        # Binarizar predições
        full_pred_binary = (full_pred > 0.5).astype(np.float32)
        full_mask_binary = (full_mask > 0.5).astype(np.float32)
        
        # Calcular Dice por classe
        for class_idx, class_name in enumerate(['exudates', 'haemorrhages']):
            pred_class = torch.from_numpy(full_pred_binary[class_idx:class_idx+1]).unsqueeze(0)
            mask_class = torch.from_numpy(full_mask_binary[class_idx:class_idx+1]).unsqueeze(0)
            
            dice = calculate_dice_score(pred_class, mask_class).mean().item()
            
            if class_idx == 0:
                all_dice_exudates.append(dice)
            else:
                all_dice_haemorrhages.append(dice)
    
    # Métricas finais
    mean_dice_exudates = np.mean(all_dice_exudates)
    mean_dice_haemorrhages = np.mean(all_dice_haemorrhages)
    mean_dice_overall = (mean_dice_exudates + mean_dice_haemorrhages) / 2
    
    print("\n" + "="*80)
    print("RESULTADOS FINAIS NO TEST SET")
    print("="*80)
    print(f"Exsudatos:    Dice = {mean_dice_exudates:.4f}")
    print(f"Hemorragias:  Dice = {mean_dice_haemorrhages:.4f}")
    print(f"Overall:      Dice = {mean_dice_overall:.4f}")
    print("="*80)
    
    return {
        'exudates_dice': mean_dice_exudates,
        'haemorrhages_dice': mean_dice_haemorrhages,
        'overall_dice': mean_dice_overall
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Função principal do experimento."""
    
    # Criar configuração usando a função dedicada
    config = create_config()
    
    print("="*80)
    print(f"EXPERIMENTO: {config.experiment_name}")
    print("Patch-Based Training com Controle Rigoroso de Leakage")
    print("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Mostrar configurações
    print(f"\n--- Configuração ---")
    print(f"Patch Size:     {config.patch_size}x{config.patch_size}")
    print(f"Overlap:        {config.patch_overlap}px")
    print(f"Batch Size:     {config.batch_size}")
    print(f"Epochs:         {config.num_epochs}")
    print(f"Learning Rate:  {config.learning_rate}")
    print(f"Encoder:        {config.encoder_name}")
    print(f"CLAHE:          {config.apply_clahe}")
    
    set_seed(config.random_state)
    
    # ==== 1. CARREGAR DADOS ====
    print("\n[1/5] Carregando dados...")
    
    data_factory = DataFactory(config)
    train_df, test_df = data_factory.create_metadata_dataframe()
    
    print(f"  Train: {len(train_df)} imagens")
    print(f"  Test:  {len(test_df)} imagens")
    
    # Estatísticas de patches
    train_stats = calculate_patches_stats(train_df, config.patch_size, config.patch_overlap)
    print(f"\n  Patches por imagem: {train_stats['patches_per_image']} (grid {train_stats['grid_size']})")
    print(f"  Total patches treino: ~{train_stats['total_patches']}")
    
    # ==== 2. CARREGAR SPLITS FROZEN ====
    print("\n[2/5] Carregando splits frozen...")
    
    splits_data = load_frozen_splits(config)
    
    # ==== 3. CROSS-VALIDATION ====
    print("\n[3/5] Iniciando Cross-Validation...")
    print("="*80)
    
    cv_results = []
    checkpoint_paths = []
    
    trainer = PatchTrainer(config, device)
    
    for fold_idx, fold_data in enumerate(splits_data['folds']):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1}/{len(splits_data['folds'])}")
        print(f"{'='*80}")
        
        # Separar train/val por IMAGENS
        train_indices = fold_data['train_indices']
        val_indices = fold_data['val_indices']
        
        train_fold_df = train_df.iloc[train_indices].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_indices].reset_index(drop=True)
        
        # VERIFICAR LEAKAGE
        verify_no_leakage(train_fold_df, val_fold_df, fold_idx + 1)
        
        # Criar datasets de PATCHES (a partir das imagens separadas)
        train_dataset = ROPDatasetPatches(
            dataframe=train_fold_df,
            config=config,
            patch_size=config.patch_size,
            overlap=config.patch_overlap,
            is_train=True
        )
        
        val_dataset = ROPDatasetPatches(
            dataframe=val_fold_df,
            config=config,
            patch_size=config.patch_size,
            overlap=config.patch_overlap,
            is_train=False
        )
        
        print(f"\n  Train: {len(train_fold_df)} imagens → {len(train_dataset)} patches")
        print(f"  Val:   {len(val_fold_df)} imagens → {len(val_dataset)} patches")
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # Treinar fold
        trainer.reset_for_fold()
        fold_result = trainer.train_fold(train_loader, val_loader, fold_idx, checkpoint_dir)
        
        cv_results.append(fold_result)
        checkpoint_paths.append(fold_result['best_model_path'])
        
        print(f"\n  Fold {fold_idx + 1} - Best Val Dice: {fold_result['best_val_dice']:.4f}")
    
    # ==== 4. RESULTADOS CV ====
    print("\n[4/5] Resultados da Cross-Validation")
    print("="*80)
    
    all_dice = [r['best_val_dice'] for r in cv_results]
    mean_dice = np.mean(all_dice)
    std_dice = np.std(all_dice)
    
    print(f"\nFolds individuais:")
    for i, dice in enumerate(all_dice):
        print(f"  Fold {i+1}: {dice:.4f}")
    
    print(f"\nMédia: {mean_dice:.4f} ± {std_dice:.4f}")
    
    # ==== 5. AVALIAR NO TEST SET ====
    print("\n[5/5] Avaliando no Test Set...")
    
    test_results = evaluate_on_test(trainer, test_df, config, checkpoint_paths)
    
    # ==== SALVAR RESULTADOS ====
    results = {
        'experiment': config.experiment_name,
        'config': {
            'patch_size': config.patch_size,
            'overlap': config.patch_overlap,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'learning_rate': config.learning_rate,
            'encoder': config.encoder_name,
            'apply_clahe': config.apply_clahe
        },
        'cv_results': {
            'folds': all_dice,
            'mean': mean_dice,
            'std': std_dice
        },
        'test_results': test_results
    }
    
    results_path = os.path.join(config.output_dir, f'{config.experiment_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Resultados salvos em: {results_path}")
    
    # ==== COMPARAÇÃO COM BASELINE ====
    print("\n" + "="*80)
    print("COMPARAÇÃO COM BASELINE")
    print("="*80)
    
    baseline_dice = 0.6721  # Wavelet Skip 1
    diff = test_results['overall_dice'] - baseline_dice
    diff_pct = (diff / baseline_dice) * 100
    
    print(f"Baseline (Wavelet Skip 1): {baseline_dice:.4f}")
    print(f"Patch-Based:               {test_results['overall_dice']:.4f}")
    print(f"Diferença:                 {diff:+.4f} ({diff_pct:+.2f}%)")
    
    if diff > 0:
        print("\n🎉 NOVO MELHOR RESULTADO!")
    else:
        print("\n⚠️ Não superou o baseline")
    
    print("\n" + "="*80)
    print("EXPERIMENTO CONCLUÍDO")
    print("="*80)


if __name__ == "__main__":
    main()
