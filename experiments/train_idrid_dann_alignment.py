"""
FASE 1: Domain Adversarial Neural Network (DANN) para Alignment
Source: ORIGA (classificação de glaucoma) 
Target: IDRID (segmentação de exsudatos/hemorragias)

Objetivo: Treinar um encoder EfficientNet-B0 que:
1. Aprende features gerais de fundoscopia
2. Alinha as distribuições ORIGA e IDRID
3. Será usado posteriormente como encoder de U-Net para segmentação
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Importações locais
from data_factory.data_factory import DataFactory
from data_factory.RetinaDataset import RetinaDataset

# ========================
# 1. GRADIENT REVERSAL LAYER
# ========================

class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalFn.apply(x, alpha)

# ========================
# 2. MODELO DANN
# ========================

class EfficientNetDANN(nn.Module):
    """
    EfficientNet-B0 com classificador de domínio adversarial
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetDANN, self).__init__()
        
        # Backbone EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        n_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Identity()
        
        # Classificador de Classe (para ORIGA - glaucoma)
        self.class_classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

        # Classificador de Domínio (ORIGA vs IDRID)
        self.domain_classifier = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Binary: 0=ORIGA, 1=IDRID
        )

    def forward(self, x, alpha=1.0):
        # Feature extraction
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)

        # Class prediction
        class_output = self.class_classifier(features)

        # Domain prediction com gradient reversal
        reverse_features = grad_reverse(features, alpha)
        domain_output = self.domain_classifier(reverse_features)

        return class_output, domain_output, features  # Retornamos features também

# ========================
# 3. UTILIDADES
# ========================

def get_alpha(current_step, total_steps):
    """Schedule progressivo do alpha (0 -> 1)"""
    p = float(current_step) / total_steps
    return 2. / (1. + np.exp(-10 * p)) - 1

def mixup_data(x1, x2, alpha=1.0):
    """MixUp entre domínios"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    mixed_x = lam * x1 + (1 - lam) * x2
    return mixed_x, lam

def infinite_iterable(loader):
    """Iterable infinito para dataloaders de tamanhos diferentes"""
    while True:
        for batch in loader:
            yield batch

# ========================
# 4. DATASET IDRID (SEM LABELS - só precisamos das imagens)
# ========================

class IDRIDDataset(torch.utils.data.Dataset):
    """
    Dataset simples para IDRID - apenas carrega imagens
    Não precisa de labels nesta fase, pois é apenas target domain
    """
    def __init__(self, img_paths, transform=None, domain_label=1):
        self.img_paths = img_paths
        self.transform = transform
        self.domain_label = domain_label
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.img_paths[idx]).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Label dummy para manter compatibilidade
        label = torch.tensor(0, dtype=torch.long)
        domain = torch.tensor(self.domain_label, dtype=torch.float32)
        
        return img, label, domain

# ========================
# 5. TREINAMENTO DANN
# ========================

def train_dann_alignment(
    source_dataset,
    target_dataset,
    num_epochs=50,
    batch_size=16,
    lr=1e-4,
    device='cuda',
    save_dir='outputs/dann_idrid_origa'
):
    """
    Treina DANN para alinhar ORIGA (source) e IDRID (target)
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"DANN ALIGNMENT: ORIGA → IDRID")
    print(f"{'='*60}")
    print(f"Source (ORIGA): {len(source_dataset)} images")
    print(f"Target (IDRID): {len(target_dataset)} images")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Dataloaders
    source_loader = DataLoader(source_dataset, batch_size=batch_size, 
                              shuffle=True, drop_last=True, num_workers=4)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, 
                              shuffle=True, drop_last=True, num_workers=4)
    
    # Iterators infinitos
    source_iter = infinite_iterable(source_loader)
    target_iter = infinite_iterable(target_loader)
    
    steps_per_epoch = min(len(source_loader), len(target_loader))
    total_steps = num_epochs * steps_per_epoch
    
    # Modelo
    model = EfficientNetDANN(num_classes=2, pretrained=True).to(device)
    
    # Otimizador
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Loss functions
    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.BCEWithLogitsLoss()
    
    # História
    history = {
        'class_loss': [],
        'domain_loss': [],
        'total_loss': [],
        'alpha_values': []
    }
    
    global_step = 0
    model.train()
    
    for epoch in range(num_epochs):
        running_class_loss = 0.0
        running_domain_loss = 0.0
        running_total_loss = 0.0
        
        for step in range(steps_per_epoch):
            global_step += 1
            alpha = get_alpha(global_step, total_steps)
            
            # Batch source e target
            img_s, label_s, _ = next(source_iter)
            img_t, _, _ = next(target_iter)
            
            img_s = img_s.to(device)
            img_t = img_t.to(device)
            label_s = label_s.to(device)
            
            # Domain labels: 0=source, 1=target
            domain_y_s = torch.zeros(img_s.size(0), 1).to(device)
            domain_y_t = torch.ones(img_t.size(0), 1).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - Source (temos labels)
            class_preds_s, domain_preds_s, _ = model(img_s, alpha=alpha)
            loss_class = criterion_class(class_preds_s, label_s)
            loss_dom_s = criterion_domain(domain_preds_s, domain_y_s)
            
            # Forward pass - Target (sem labels de classe)
            _, domain_preds_t, _ = model(img_t, alpha=alpha)
            loss_dom_t = criterion_domain(domain_preds_t, domain_y_t)
            
            # MixUp entre domínios
            img_mix, lam = mixup_data(img_s, img_t, alpha=1.0)
            mixed_domain_label = (1 - lam) * torch.ones(img_s.size(0), 1).to(device)
            _, domain_preds_mix, _ = model(img_mix, alpha=alpha)
            loss_dom_mix = criterion_domain(domain_preds_mix, mixed_domain_label)
            
            # Loss combinada
            loss_domain_combined = (loss_dom_s + loss_dom_t + loss_dom_mix)
            loss_total = loss_class + loss_domain_combined * 0.5
            
            loss_total.backward()
            optimizer.step()
            
            running_class_loss += loss_class.item()
            running_domain_loss += loss_domain_combined.item()
            running_total_loss += loss_total.item()
        
        # Médias por época
        avg_class = running_class_loss / steps_per_epoch
        avg_domain = running_domain_loss / steps_per_epoch
        avg_total = running_total_loss / steps_per_epoch
        
        history['class_loss'].append(avg_class)
        history['domain_loss'].append(avg_domain)
        history['total_loss'].append(avg_total)
        history['alpha_values'].append(alpha)
        
        print(f"Epoch [{epoch+1:03d}/{num_epochs}] "
              f"Class: {avg_class:.4f} | "
              f"Domain: {avg_domain:.4f} | "
              f"Total: {avg_total:.4f} | "
              f"Alpha: {alpha:.3f}")
        
        # Salvar checkpoint a cada 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(save_dir) / f'dann_checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, checkpoint_path)
    
    # Salvar modelo final
    final_model_path = Path(save_dir) / 'dann_encoder_final.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'source_size': len(source_dataset),
            'target_size': len(target_dataset)
        }
    }, final_model_path)
    
    print(f"\n✓ Modelo salvo em: {final_model_path}")
    
    # Plot losses
    plot_training_curves(history, save_dir)
    
    return model, history

def plot_training_curves(history, save_dir):
    """Plota curvas de treinamento"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    epochs = range(1, len(history['total_loss']) + 1)
    
    # Loss de classificação e domínio
    axes[0].plot(epochs, history['class_loss'], label='Class Loss', linewidth=2)
    axes[0].plot(epochs, history['domain_loss'], label='Domain Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Class vs Domain Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss total e alpha
    ax1 = axes[1]
    ax2 = ax1.twinx()
    
    ax1.plot(epochs, history['total_loss'], 'b-', label='Total Loss', linewidth=2)
    ax2.plot(epochs, history['alpha_values'], 'r--', label='Alpha', linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss', color='b')
    ax2.set_ylabel('Alpha', color='r')
    ax1.set_title('Total Loss & Alpha Schedule')
    
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'training_curves.png', dpi=150)
    plt.close()
    
    print(f"✓ Gráficos salvos em: {save_dir}/training_curves.png")

# ========================
# 6. MAIN
# ========================

if __name__ == "__main__":
    # Configurações
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMG_SIZE = 256
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LR = 1e-4
    
    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # ========================
    # SOURCE: ORIGA
    # ========================
    print("Carregando ORIGA (source)...")
    origa_factory = DataFactory(
        img_path='/backup/lucas/datasets/origa/ORIGA/Images',
        metadata_path='/backup/lucas/datasets/origa/ORIGA/OrigaList.csv'
    )
    origa_df = origa_factory.load_data(verbose=True)
    source_dataset = RetinaDataset(
        dataframe=origa_df, 
        transform=train_transforms, 
        domain_label=0
    )
    
    # ========================
    # TARGET: IDRID
    # ========================
    print("\nCarregando IDRID (target)...")
    idrid_img_dir = Path('/backup/lucas/datasets/IDRID/1. Original Images/a. Training Set')
    idrid_img_paths = sorted(list(idrid_img_dir.glob('*.jpg')))
    
    if len(idrid_img_paths) == 0:
        print("⚠ ATENÇÃO: Nenhuma imagem encontrada em IDRID!")
        print(f"   Verificar caminho: {idrid_img_dir}")
    else:
        print(f"✓ {len(idrid_img_paths)} imagens encontradas em IDRID")
    
    target_dataset = IDRIDDataset(
        img_paths=idrid_img_paths,
        transform=train_transforms,
        domain_label=1
    )
    
    # ========================
    # TREINAMENTO
    # ========================
    model, history = train_dann_alignment(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        device=DEVICE,
        save_dir='outputs/dann_idrid_origa'
    )
    
    print("\n" + "="*60)
    print("✓ FASE 1 (DANN Alignment) CONCLUÍDA!")
    print("="*60)
    print("\nPróximos passos:")
    print("1. Execute o script de Fine-tuning para Segmentação")
    print("2. O encoder treinado será carregado e acoplado em uma U-Net")
    print("3. Treinar apenas com as 54 imagens do IDRID para segmentação")
