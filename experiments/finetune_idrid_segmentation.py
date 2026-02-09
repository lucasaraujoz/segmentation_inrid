"""
FASE 2: Fine-tuning para Segmentação com Encoder Pré-treinado via DANN

Pega o encoder EfficientNet-B0 treinado na Fase 1 (DANN alignment)
e acopla em uma U-Net para segmentar exsudatos e hemorragias no IDRID.

Arquitetura:
- Encoder: EfficientNet-B0 (features da Fase 1 - conhece estruturas de retina)
- Decoder: U-Net style com skip connections
- Output: 2 canais (exsudato, hemorragia) ou 3 se incluir background
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# ========================
# 1. U-NET COM EFFICIENTNET ENCODER
# ========================

class EfficientNetUNet(nn.Module):
    """
    U-Net com EfficientNet-B0 como encoder
    Permite carregar pesos pré-treinados do DANN
    """
    def __init__(self, num_classes=3, pretrained_dann_path=None):
        super(EfficientNetUNet, self).__init__()
        
        # Encoder: EfficientNet-B0
        if pretrained_dann_path:
            # Carregar do DANN
            print(f"Carregando encoder de: {pretrained_dann_path}")
            self.encoder = self._load_dann_encoder(pretrained_dann_path)
        else:
            # ImageNet weights
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.encoder = models.efficientnet_b0(weights=weights).features
        
        # Feature channels em cada estágio do EfficientNet-B0
        # [16, 24, 40, 112, 1280]
        encoder_channels = [16, 24, 40, 112, 1280]
        
        # Decoder com skip connections
        self.decoder4 = self._make_decoder_block(encoder_channels[4], encoder_channels[3])
        self.decoder3 = self._make_decoder_block(encoder_channels[3], encoder_channels[2])
        self.decoder2 = self._make_decoder_block(encoder_channels[2], encoder_channels[1])
        self.decoder1 = self._make_decoder_block(encoder_channels[1], encoder_channels[0])
        
        # Final upsampling + classificação
        self.final_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[0], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
        # Guardar índices dos estágios do encoder
        # EfficientNet-B0 tem 9 blocos, pegamos saídas estratégicas
        self.encoder_indices = [2, 3, 5, 7, 8]  # Outputs de cada stage
    
    def _load_dann_encoder(self, checkpoint_path):
        """Carrega apenas o encoder do modelo DANN"""
        from models import EfficientNetDANN  # Importar aqui para evitar circular
        
        # Criar modelo DANN
        dann_model = models.efficientnet_b0(weights=None)
        
        # Carregar checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extrair state_dict do encoder
        state_dict = checkpoint['model_state_dict']
        
        # Filtrar apenas as chaves do backbone
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.features'):
                # Remover 'backbone.' do nome
                new_key = k.replace('backbone.', '')
                encoder_state[new_key] = v
        
        # Carregar pesos
        dann_model.features.load_state_dict(encoder_state, strict=False)
        print(f"✓ {len(encoder_state)} parâmetros carregados do DANN encoder")
        
        return dann_model.features
    
    def _make_decoder_block(self, in_channels, out_channels):
        """Bloco do decoder com upsampling + convs"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Input: (B, 3, H, W)
        B, _, H, W = x.shape
        
        # Encoder com skip connections
        skip_connections = []
        
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            if idx in self.encoder_indices[:-1]:  # Salvar skips (exceto último)
                skip_connections.append(x)
        
        # x agora é a saída do encoder (features mais profundas)
        
        # Decoder com skip connections
        x = self.decoder4(x)  # + skip_connections[3] se quiser concatenar
        x = self.decoder3(x)
        x = self.decoder2(x)
        x = self.decoder1(x)
        
        # Upsampling final para tamanho original
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        # Classificação por pixel
        x = self.final_conv(x)
        
        return x  # (B, num_classes, H, W)

# ========================
# 2. DATASET IDRID SEGMENTAÇÃO
# ========================

class IDRIDSegmentationDataset(Dataset):
    """
    Dataset para segmentação de Exsudatos e Hemorragias no IDRID
    """
    def __init__(self, image_paths, mask_ex_paths, mask_he_paths, 
                 transform=None, img_size=256):
        self.image_paths = image_paths
        self.mask_ex_paths = mask_ex_paths
        self.mask_he_paths = mask_he_paths
        self.transform = transform
        self.img_size = img_size
        
        # Transforms para máscaras (sem augmentation, apenas resize)
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), 
                            interpolation=transforms.InterpolationMode.NEAREST)
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Carregar imagem
        img = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Carregar máscaras
        mask_ex = Image.open(self.mask_ex_paths[idx]).convert('L')
        mask_he = Image.open(self.mask_he_paths[idx]).convert('L')
        
        # Aplicar transforms
        if self.transform:
            img = self.transform(img)
        
        mask_ex = self.mask_transform(mask_ex)
        mask_he = self.mask_transform(mask_he)
        
        # Converter para tensor e binarizar
        mask_ex = torch.from_numpy(np.array(mask_ex) > 127).long()
        mask_he = torch.from_numpy(np.array(mask_he) > 127).long()
        
        # Criar máscara multiclass: 0=bg, 1=exsudato, 2=hemorragia
        mask = torch.zeros_like(mask_ex)
        mask[mask_ex == 1] = 1
        mask[mask_he == 1] = 2
        
        return img, mask

# ========================
# 3. LOSSES E MÉTRICAS
# ========================

class DiceLoss(nn.Module):
    """Dice Loss para segmentação"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # pred: (B, C, H, W) logits
        # target: (B, H, W) class indices
        
        num_classes = pred.size(1)
        pred = F.softmax(pred, dim=1)
        
        # One-hot encoding do target
        target_onehot = F.one_hot(target, num_classes=num_classes)  # (B, H, W, C)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Flatten
        pred = pred.contiguous().view(-1)
        target_onehot = target_onehot.contiguous().view(-1)
        
        intersection = (pred * target_onehot).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target_onehot.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combina CrossEntropy + Dice"""
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

def dice_coefficient(pred, target, num_classes=3):
    """Calcula Dice Score por classe"""
    pred = torch.argmax(pred, dim=1)  # (B, H, W)
    
    dice_scores = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union == 0:
            dice_scores.append(1.0)  # Perfeito se não existem amostras
        else:
            dice_scores.append((2. * intersection / union).item())
    
    return dice_scores

# ========================
# 4. TREINAMENTO
# ========================

def train_segmentation(
    train_dataset,
    val_dataset,
    pretrained_dann_path=None,
    num_epochs=100,
    batch_size=8,
    lr=1e-3,
    device='cuda',
    save_dir='outputs/idrid_segmentation'
):
    """
    Fine-tuning para segmentação usando encoder do DANN
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"FINE-TUNING SEGMENTAÇÃO IDRID")
    print(f"{'='*60}")
    print(f"Train: {len(train_dataset)} imagens")
    print(f"Val: {len(val_dataset)} imagens")
    print(f"Encoder pré-treinado: {pretrained_dann_path is not None}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    
    # Modelo
    model = EfficientNetUNet(
        num_classes=3,  # bg, exsudato, hemorragia
        pretrained_dann_path=pretrained_dann_path
    ).to(device)
    
    # Otimizador com learning rates diferentes
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': lr * 0.1},  # Encoder já treinado
        {'params': decoder_params, 'lr': lr}         # Decoder do zero
    ], weight_decay=1e-5)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Loss
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    
    # História
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice_bg': [],
        'val_dice_ex': [],
        'val_dice_he': [],
        'val_dice_mean': []
    }
    
    best_dice = 0.0
    
    for epoch in range(num_epochs):
        # ===== TREINO =====
        model.train()
        running_loss = 0.0
        
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_train_loss = running_loss / len(train_loader)
        
        # ===== VALIDAÇÃO =====
        model.eval()
        val_loss = 0.0
        all_dice_scores = {'bg': [], 'ex': [], 'he': []}
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Dice por classe
                dice_scores = dice_coefficient(outputs, masks, num_classes=3)
                all_dice_scores['bg'].append(dice_scores[0])
                all_dice_scores['ex'].append(dice_scores[1])
                all_dice_scores['he'].append(dice_scores[2])
        
        epoch_val_loss = val_loss / len(val_loader)
        
        # Médias dos Dice Scores
        dice_bg = np.mean(all_dice_scores['bg'])
        dice_ex = np.mean(all_dice_scores['ex'])
        dice_he = np.mean(all_dice_scores['he'])
        dice_mean = np.mean([dice_ex, dice_he])  # Média sem background
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_dice_bg'].append(dice_bg)
        history['val_dice_ex'].append(dice_ex)
        history['val_dice_he'].append(dice_he)
        history['val_dice_mean'].append(dice_mean)
        
        print(f"Epoch [{epoch+1:03d}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Dice - EX: {dice_ex:.3f}, HE: {dice_he:.3f}, Mean: {dice_mean:.3f}")
        
        # Salvar melhor modelo
        if dice_mean > best_dice:
            best_dice = dice_mean
            best_model_path = Path(save_dir) / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice_mean': dice_mean,
                'history': history
            }, best_model_path)
            print(f"  ✓ Melhor modelo salvo! Dice Mean: {dice_mean:.4f}")
        
        scheduler.step()
    
    # Salvar modelo final
    final_path = Path(save_dir) / 'final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history
    }, final_path)
    
    # Plot
    plot_segmentation_curves(history, save_dir)
    
    print(f"\n✓ Treinamento concluído!")
    print(f"  Melhor Dice Mean: {best_dice:.4f}")
    
    return model, history

def plot_segmentation_curves(history, save_dir):
    """Plota curvas de treinamento"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Losses
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice Scores
    axes[1].plot(epochs, history['val_dice_ex'], label='Exsudato', linewidth=2)
    axes[1].plot(epochs, history['val_dice_he'], label='Hemorragia', linewidth=2)
    axes[1].plot(epochs, history['val_dice_mean'], label='Mean', linewidth=2, linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Validation Dice Scores')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'training_curves.png', dpi=150)
    plt.close()

# ========================
# 5. MAIN
# ========================

if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMG_SIZE = 256
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LR = 1e-3
    
    # Caminho do encoder pré-treinado (DANN)
    DANN_ENCODER_PATH = 'outputs/dann_idrid_origa/dann_encoder_final.pth'
    
    # Verificar se existe
    if not Path(DANN_ENCODER_PATH).exists():
        print(f"⚠ AVISO: Encoder DANN não encontrado em {DANN_ENCODER_PATH}")
        print("  Iniciando com pesos ImageNet")
        DANN_ENCODER_PATH = None
    
    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Carregar dados IDRID
    base_path = Path('/backup/lucas/datasets/IDRID')
    
    # Imagens
    train_imgs = sorted((base_path / '1. Original Images/a. Training Set').glob('*.jpg'))
    test_imgs = sorted((base_path / '1. Original Images/b. Testing Set').glob('*.jpg'))
    
    # Máscaras Exsudatos
    train_masks_ex = sorted((base_path / '2. All Segmentation Groundtruths/a. Training Set/1. Hard Exudates').glob('*_EX.tif'))
    test_masks_ex = sorted((base_path / '2. All Segmentation Groundtruths/b. Testing Set/1. Hard Exudates').glob('*_EX.tif'))
    
    # Máscaras Hemorragias
    train_masks_he = sorted((base_path / '2. All Segmentation Groundtruths/a. Training Set/2. Haemorrhages').glob('*_HE.tif'))
    test_masks_he = sorted((base_path / '2. All Segmentation Groundtruths/b. Testing Set/2. Haemorrhages').glob('*_HE.tif'))
    
    print(f"Train: {len(train_imgs)} imagens, {len(train_masks_ex)} EX, {len(train_masks_he)} HE")
    print(f"Test: {len(test_imgs)} imagens, {len(test_masks_ex)} EX, {len(test_masks_he)} HE")
    
    # Criar datasets
    train_dataset = IDRIDSegmentationDataset(
        train_imgs, train_masks_ex, train_masks_he,
        transform=train_transforms, img_size=IMG_SIZE
    )
    
    val_dataset = IDRIDSegmentationDataset(
        test_imgs, test_masks_ex, test_masks_he,
        transform=val_transforms, img_size=IMG_SIZE
    )
    
    # Treinar
    model, history = train_segmentation(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        pretrained_dann_path=DANN_ENCODER_PATH,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        device=DEVICE
    )
