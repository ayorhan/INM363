"""
Perceptual Losses for Real-Time Style Transfer and Super-Resolution

Source: https://arxiv.org/abs/1603.08155
Type: CNN (Convolutional Neural Network)

Architecture:
- Feed-forward transformation network
- VGG-based perceptual loss
- Residual blocks
- Instance normalization
- Skip connections

Pros:
- Real-time performance
- Single pass inference
- Stable results
- Memory efficient

Cons:
- Single style per model
- Training required for each style
- Limited adaptability
- Fixed style strength
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Tuple

class ResidualBlock(nn.Module):
    """Enhanced residual block with improved normalization and activation"""
    def __init__(self, channels: int, use_dropout: bool = False):
        super(ResidualBlock, self).__init__()
        
        layers = [
            nn.ReflectionPad2d(1),  # Better border handling
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True)
        ]
        
        if use_dropout:
            layers.append(nn.Dropout(0.5))
            
        layers.extend([
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels, affine=True)
        ])
        
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class JohnsonModel(nn.Module):
    """
    Enhanced Johnson model with improved architecture and features
    """
    def __init__(self, input_channels: int = 3, 
                 base_filters: int = 64,
                 n_residuals: int = 5,
                 use_dropout: bool = True):
        super(JohnsonModel, self).__init__()

        # Initial convolution with reflection padding
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(4),  # Better border handling
            nn.Conv2d(input_channels, base_filters, kernel_size=9),
            nn.InstanceNorm2d(base_filters, affine=True),
            nn.ReLU(inplace=True)
        )

        # Downsampling with skip connections
        self.down_blocks = nn.ModuleList([
            self._build_down_block(base_filters, base_filters * 2),
            self._build_down_block(base_filters * 2, base_filters * 4)
        ])

        # Enhanced residual blocks with dropout option
        self.residuals = nn.ModuleList([
            ResidualBlock(base_filters * 4, use_dropout)
            for _ in range(n_residuals)
        ])

        # Upsampling with skip connections
        self.up_blocks = nn.ModuleList([
            self._build_up_block(base_filters * 4, base_filters * 2),
            self._build_up_block(base_filters * 2, base_filters)
        ])

        # Output convolution
        self.output = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(base_filters, input_channels, kernel_size=9),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _build_down_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Build a downsampling block with improved architecture"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=2, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def _build_up_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Build an upsampling block with improved architecture"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                              kernel_size=3, stride=2,
                              padding=1, output_padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self, m: nn.Module):
        """Initialize network weights with improved method"""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                  nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial features
        x = self.initial(x)
        
        # Store skip connections
        skip_connections = []
        
        # Downsampling with skip connections
        for down_block in self.down_blocks:
            skip_connections.append(x)
            x = down_block(x)
        
        # Residual blocks
        for res_block in self.residuals:
            x = res_block(x)
        
        # Upsampling with skip connections
        for up_block, skip in zip(self.up_blocks, 
                                reversed(skip_connections)):
            x = up_block(x)
            x = torch.cat([x, skip], dim=1)
        
        return self.output(x)

class EnhancedPerceptualLoss(nn.Module):
    """
    Enhanced perceptual loss with multiple VGG layers and style loss option
    """
    def __init__(self, 
                 content_layers: List[str] = ['relu4_1'],
                 style_layers: List[str] = ['relu1_1', 'relu2_1', 
                                          'relu3_1', 'relu4_1'],
                 content_weight: float = 1.0,
                 style_weight: float = 0.0):
        super(EnhancedPerceptualLoss, self).__init__()
        
        self.content_weight = content_weight
        self.style_weight = style_weight
        
        # Load VGG model
        vgg = models.vgg19(pretrained=True).features.eval()
        
        # Create layer mapping
        self.layer_mapping = {
            'relu1_1': '2',
            'relu2_1': '7',
            'relu3_1': '12',
            'relu4_1': '21',
            'relu5_1': '30'
        }
        
        # Extract required layers
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.layers = nn.ModuleDict()
        
        current_block = nn.Sequential()
        last_layer = 0
        
        for name, layer in vgg.named_children():
            current_block.add_module(name, layer)
            
            if name in [self.layer_mapping[l] for l in content_layers + style_layers]:
                self.layers[name] = current_block
                current_block = nn.Sequential()
                last_layer = int(name)
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gram matrix for style loss"""
        b, c, h, w = x.size()
        features = x.view(b, c, -1)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

    def forward(self, generated: torch.Tensor, 
                target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        content_loss = 0
        style_loss = 0
        
        # Compute losses for each layer
        for name, layer in self.layers.items():
            gen_features = layer(generated)
            target_features = layer(target)
            
            # Content loss
            if name in [self.layer_mapping[l] for l in self.content_layers]:
                content_loss += F.mse_loss(gen_features, target_features)
            
            # Style loss
            if name in [self.layer_mapping[l] for l in self.style_layers]:
                gen_gram = self.gram_matrix(gen_features)
                target_gram = self.gram_matrix(target_features)
                style_loss += F.mse_loss(gen_gram, target_gram)
        
        total_loss = (self.content_weight * content_loss + 
                     self.style_weight * style_loss)
        
        return total_loss, {
            'content_loss': content_loss.item(),
            'style_loss': style_loss.item()
        }
