"""
Wavelet-enhanced skip connection module.
Applies DWT 2D to extract high-frequency features (edges/details).
"""

import torch
import torch.nn as nn
import pywt
import numpy as np


class WaveletSkipConnection(nn.Module):
    """
    Aplica DWT 2D no skip e concatena coeficientes de alta frequência.
    
    Args:
        in_channels: Número de canais do skip original
        wavelet: Tipo de wavelet ('haar' ou 'db1')
    """
    
    def __init__(self, in_channels, wavelet='haar'):
        super().__init__()
        self.in_channels = in_channels
        self.wavelet = wavelet
        
        # Conv 1x1 para reduzir canais após concatenação
        # skip (C) + LH (C) + HL (C) + HH (C) = 4C → C
        self.channel_reduction = nn.Conv2d(
            in_channels * 4, 
            in_channels, 
            kernel_size=1, 
            bias=False
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Args:
            x: Skip tensor [B, C, H, W]
        
        Returns:
            Enhanced skip [B, C, H, W]
        """
        batch_size, channels, height, width = x.shape
        
        # Aplicar DWT 2D em cada canal
        wavelet_features = []
        
        for b in range(batch_size):
            batch_wavelets = []
            
            for c in range(channels):
                # Extrair canal único
                channel_data = x[b, c].detach().cpu().numpy()
                
                # DWT 2D nível 1
                coeffs = pywt.dwt2(channel_data, self.wavelet)
                cA, (cH, cV, cD) = coeffs
                
                # Guardar coeficientes de alta frequência
                # LH (horizontal), HL (vertical), HH (diagonal)
                batch_wavelets.append([cH, cV, cD])
            
            # Stack todos os canais
            # LH, HL, HH para cada canal
            LH = np.stack([w[0] for w in batch_wavelets], axis=0)  # [C, H/2, W/2]
            HL = np.stack([w[1] for w in batch_wavelets], axis=0)
            HH = np.stack([w[2] for w in batch_wavelets], axis=0)
            
            wavelet_features.append([LH, HL, HH])
        
        # Converter para tensor
        LH_batch = torch.from_numpy(
            np.stack([w[0] for w in wavelet_features], axis=0)
        ).float().to(x.device)
        
        HL_batch = torch.from_numpy(
            np.stack([w[1] for w in wavelet_features], axis=0)
        ).float().to(x.device)
        
        HH_batch = torch.from_numpy(
            np.stack([w[2] for w in wavelet_features], axis=0)
        ).float().to(x.device)
        
        # Upsample wavelets para mesma resolução do skip
        # Wavelets são H/2, W/2 → interpolar para H, W
        LH_up = nn.functional.interpolate(
            LH_batch, size=(height, width), mode='bilinear', align_corners=False
        )
        HL_up = nn.functional.interpolate(
            HL_batch, size=(height, width), mode='bilinear', align_corners=False
        )
        HH_up = nn.functional.interpolate(
            HH_batch, size=(height, width), mode='bilinear', align_corners=False
        )
        
        # Concatenar: skip original + LH + HL + HH
        enhanced = torch.cat([x, LH_up, HL_up, HH_up], dim=1)  # [B, 4C, H, W]
        
        # Reduzir canais de volta
        enhanced = self.channel_reduction(enhanced)  # [B, C, H, W]
        enhanced = self.bn(enhanced)
        enhanced = self.relu(enhanced)
        
        return enhanced
