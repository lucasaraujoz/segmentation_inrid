"""
UNet com wavelet enhancement apenas no primeiro skip connection.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from models.wavelet_skip import WaveletSkipConnection


class UnetWaveletSkip1(nn.Module):
    """
    UNet do SMP com wavelet no primeiro skip.
    Usa hook para interceptar e modificar o skip connection.
    
    Args:
        encoder_name: Nome do encoder (ex: 'efficientnet-b4')
        encoder_weights: Pesos pré-treinados ('imagenet' ou None)
        in_channels: Canais de entrada (3 para RGB)
        classes: Número de classes de saída
    """
    
    def __init__(
        self, 
        encoder_name='efficientnet-b4',
        encoder_weights='imagenet',
        in_channels=3,
        classes=2
    ):
        super().__init__()
        
        # Base UNet do SMP
        self.base_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None
        )
        
        # Número de canais do primeiro skip (features[1])
        if 'efficientnet-b4' in encoder_name:
            skip1_channels = 48
        elif 'efficientnet-b3' in encoder_name:
            skip1_channels = 40
        elif 'efficientnet-b5' in encoder_name:
            skip1_channels = 48
        else:
            skip1_channels = 48
        
        # Módulo wavelet para primeiro skip
        self.wavelet_skip1 = WaveletSkipConnection(
            in_channels=skip1_channels,
            wavelet='haar'
        )
        
        # Registrar hook para aplicar wavelet
        self.register_hooks()
    
    def forward(self, x):
        """
        Forward pass simples - deixa o modelo fazer tudo.
        """
        # Usar o forward normal do modelo base
        # Vamos aplicar wavelet através de um wrapper no encoder
        return self.base_model(x)
    
    def _custom_forward_hook(self, module, input, output):
        """
        Hook para modificar output do encoder stage 1.
        """
        # Output do encoder é uma tupla de features
        if isinstance(output, (list, tuple)):
            features = list(output)
            # Aplicar wavelet no features[1] (primeiro skip real)
            if len(features) > 1:
                features[1] = self.wavelet_skip1(features[1])
            return tuple(features) if isinstance(output, tuple) else features
        return output
    
    def register_hooks(self):
        """
        Registrar hook no encoder para modificar primeiro skip.
        """
        # Hook no encoder completo
        self.base_model.encoder.register_forward_hook(self._custom_forward_hook)
