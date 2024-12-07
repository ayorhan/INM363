"""
CycleGAN: Unpaired Image-to-Image Translation

Source: https://arxiv.org/abs/1703.10593
Type: GAN (Generative Adversarial Network)

Architecture:
- Dual generators and discriminators
- Cycle consistency loss
- Identity mapping
- PatchGAN discrimination
- ResNet-based generators

Pros:
- Works with unpaired data
- Bidirectional translation
- Preserves image structure
- Versatile applications

Cons:
- Mode collapse risk
- Training instability
- Limited by cycle consistency
- Can produce artifacts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_dropout=False):
        super(ResidualBlock, self).__init__()
        
        # Enhanced residual block with optional dropout
        layers = [
            nn.ReflectionPad2d(1),  # Better border handling than zero padding
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True)
        ]
        
        if use_dropout:
            layers.append(nn.Dropout(0.5))
            
        layers.extend([
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels)
        ])
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, n_residuals=9, 
                 base_filters=64, use_dropout=True):
        super(Generator, self).__init__()
        
        # Initial convolution with reflection padding
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, base_filters, kernel_size=7),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList([
            self._build_down_block(base_filters, base_filters * 2),
            self._build_down_block(base_filters * 2, base_filters * 4)
        ])
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(base_filters * 4, use_dropout)
            for _ in range(n_residuals)
        ])
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList([
            self._build_up_block(base_filters * 4, base_filters * 2),
            self._build_up_block(base_filters * 2, base_filters)
        ])
        
        # Output convolution
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_filters, output_channels, kernel_size=7),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _build_down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                     kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def _build_up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                              kernel_size=3, stride=2,
                              padding=1, output_padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Initial convolution
        x = self.initial(x)
        
        # Store skip connections
        skip_connections = []
        
        # Downsampling
        for down_block in self.down_blocks:
            skip_connections.append(x)
            x = down_block(x)
            
        # Residual blocks
        for res_block in self.residual_blocks:
            x = res_block(x)
            
        # Upsampling with skip connections
        for up_block, skip in zip(self.up_blocks, 
                                reversed(skip_connections)):
            x = up_block(x)
            x = torch.cat([x, skip], dim=1)
            
        return self.output(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, base_filters=64, n_layers=3):
        super(Discriminator, self).__init__()
        
        # Initial layer without normalization
        layers = [
            nn.Conv2d(input_channels, base_filters, 
                     kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Scaling factor for number of filters
        nf_mult = 1
        nf_mult_prev = 1
        
        # Additional layers with normalization
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
            
        # Final layers
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
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        return self.model(x)

class CycleGAN(nn.Module):
    def __init__(self, config):
        super(CycleGAN, self).__init__()
        
        # Generators
        self.G_AB = Generator(
            input_channels=config.get('input_channels', 3),
            output_channels=config.get('output_channels', 3),
            n_residuals=config.get('n_residuals', 9),
            base_filters=config.get('base_filters', 64),
            use_dropout=config.get('use_dropout', True)
        )
        
        self.G_BA = Generator(
            input_channels=config.get('input_channels', 3),
            output_channels=config.get('output_channels', 3),
            n_residuals=config.get('n_residuals', 9),
            base_filters=config.get('base_filters', 64),
            use_dropout=config.get('use_dropout', True)
        )
        
        # Discriminators
        self.D_A = Discriminator(
            input_channels=config.get('input_channels', 3),
            base_filters=config.get('base_filters', 64),
            n_layers=config.get('n_layers', 3)
        )
        
        self.D_B = Discriminator(
            input_channels=config.get('input_channels', 3),
            base_filters=config.get('base_filters', 64),
            n_layers=config.get('n_layers', 3)
        )
        
    def forward(self, x, direction='AB'):
        if direction == 'AB':
            return self.G_AB(x)
        else:
            return self.G_BA(x)
