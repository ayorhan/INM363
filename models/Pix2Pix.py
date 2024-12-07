"""
Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks

Source: https://arxiv.org/abs/1611.07004
Type: GAN (Generative Adversarial Network)

Architecture:
- U-Net generator with skip connections
- PatchGAN discriminator
- Conditional adversarial training
- Instance normalization
- Multi-scale feature processing

Pros:
- Excellent for paired image translation
- Preserves structural details
- Stable training process
- Versatile applications

Cons:
- Requires paired training data
- Can produce artifacts
- Limited by training data diversity
- May struggle with complex transformations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class UNetBlock(nn.Module):
    """U-Net block with skip connections and optional dropout"""
    def __init__(self, in_channels: int, out_channels: int, 
                 use_dropout: bool = False, use_bias: bool = True):
        super(UNetBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, 
                     stride=2, padding=1, bias=use_bias),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x

class UNetGenerator(nn.Module):
    """Enhanced U-Net generator with improved architecture"""
    def __init__(self, input_channels: int = 3, output_channels: int = 3, 
                 base_filters: int = 64):
        super(UNetGenerator, self).__init__()
        
        # Encoder
        self.down1 = UNetBlock(input_channels, base_filters, use_dropout=False)
        self.down2 = UNetBlock(base_filters, base_filters * 2)
        self.down3 = UNetBlock(base_filters * 2, base_filters * 4)
        self.down4 = UNetBlock(base_filters * 4, base_filters * 8)
        self.down5 = UNetBlock(base_filters * 8, base_filters * 8)
        self.down6 = UNetBlock(base_filters * 8, base_filters * 8)
        self.down7 = UNetBlock(base_filters * 8, base_filters * 8)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 8, base_filters * 8, 4, 2, 1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder with skip connections
        self.up1 = self._build_up_block(base_filters * 8, base_filters * 8, use_dropout=True)
        self.up2 = self._build_up_block(base_filters * 16, base_filters * 8, use_dropout=True)
        self.up3 = self._build_up_block(base_filters * 16, base_filters * 8, use_dropout=True)
        self.up4 = self._build_up_block(base_filters * 16, base_filters * 4)
        self.up5 = self._build_up_block(base_filters * 8, base_filters * 2)
        self.up6 = self._build_up_block(base_filters * 4, base_filters)
        
        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, output_channels, 
                              kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _build_up_block(self, in_channels: int, out_channels: int, 
                       use_dropout: bool = False) -> nn.Sequential:
        """Build upsampling block with optional dropout"""
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,
                              kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        
        if use_dropout:
            layers.append(nn.Dropout(0.5))
            
        return nn.Sequential(*layers)
    
    def _init_weights(self, m: nn.Module):
        """Initialize network weights"""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        
        # Bottleneck
        bottleneck = self.bottleneck(d7)
        
        # Decoder with skip connections
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        
        return self.final(torch.cat([u6, d2], dim=1))

class PatchGANDiscriminator(nn.Module):
    """Enhanced PatchGAN discriminator with improved architecture"""
    def __init__(self, input_channels: int = 6, base_filters: int = 64, 
                 n_layers: int = 3):
        super(PatchGANDiscriminator, self).__init__()
        
        # Initial layer without normalization
        layers = [
            nn.Conv2d(input_channels, base_filters, 
                     kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Additional layers with normalization
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.extend([
                nn.Conv2d(base_filters * nf_mult_prev, 
                         base_filters * nf_mult,
                         kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(base_filters * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # Final layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        layers.extend([
            nn.Conv2d(base_filters * nf_mult_prev,
                     base_filters * nf_mult,
                     kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(base_filters * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters * nf_mult, 1,
                     kernel_size=4, stride=1, padding=1)
        ])
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m: nn.Module):
        """Initialize network weights"""
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class Pix2Pix(nn.Module):
    """Complete Pix2Pix model"""
    def __init__(self, config: dict):
        super(Pix2Pix, self).__init__()
        
        # Initialize generator and discriminator
        self.generator = UNetGenerator(
            input_channels=config.get('input_channels', 3),
            output_channels=config.get('output_channels', 3),
            base_filters=config.get('base_filters', 64)
        )
        
        self.discriminator = PatchGANDiscriminator(
            input_channels=config.get('input_channels', 3) * 2,
            base_filters=config.get('base_filters', 64),
            n_layers=config.get('n_layers', 3)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning generated image and discrimination"""
        fake = self.generator(x)
        real_fake = torch.cat([x, fake], dim=1)
        return fake, self.discriminator(real_fake)
