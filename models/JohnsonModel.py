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
    def __init__(self, config):
        super().__init__()
        # Extract model parameters from config
        self.input_channels = config['input_channels']
        self.output_channels = config['output_channels']
        self.base_filters = config['base_filters']
        self.n_residuals = config['n_residuals']
        self.use_dropout = config['use_dropout']
        self.norm_type = config['norm_type']
        self.content_layers = config['content_layers']
        self.style_layers = config['style_layers']
        
        # Initial convolution block
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.input_channels, self.base_filters, kernel_size=7),
            nn.InstanceNorm2d(self.base_filters),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList([
            self._build_down_block(self.base_filters, self.base_filters * 2),
            self._build_down_block(self.base_filters * 2, self.base_filters * 4)
        ])
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(self.base_filters * 4, self.use_dropout)
            for _ in range(self.n_residuals)
        ])
        
        # Upsampling blocks with correct channel dimensions
        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=self.base_filters * 4,  # 256
                    out_channels=self.base_filters * 2,  # 128
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.InstanceNorm2d(self.base_filters * 2),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=self.base_filters * 2,  # 128
                    out_channels=self.base_filters,      # 64
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.InstanceNorm2d(self.base_filters),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Final output layer
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.base_filters, self.output_channels, kernel_size=7),
            nn.Tanh()
        )
        
        # Add gradient scaling factors
        self.register_buffer('content_scale', torch.tensor(0.1))
        self.register_buffer('style_scale', torch.tensor(5.0))
        
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

    def _build_up_block(self, in_channels, out_channels):
        """Build an upsampling block with correct channel dimensions"""
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self, m: nn.Module):
        """Modified weight initialization"""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # Use smaller initialization
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu', a=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add gradient scaling
        if self.training:
            x = x * self.content_scale
        
        # Initial features
        x = self.initial(x)
        
        # Downsampling
        for down_block in self.down_blocks:
            x = down_block(x)
        
        # Residual blocks
        for res_block in self.residual_blocks:
            x = res_block(x)
        
        # Upsampling (no skip connections)
        for up_block in self.up_blocks:
            x = up_block(x)
        
        # Final output
        if self.training:
            x = x / self.content_scale
        return self.output(x)

class EnhancedPerceptualLoss(nn.Module):
    """
    Enhanced perceptual loss with multiple VGG layers and style loss option
    """
    def __init__(self, 
                 content_layers: List[str] = ['relu3_3'],
                 style_layers: List[str] = ['relu1_1', 'relu2_1', 
                                          'relu3_1', 'relu4_1', 'relu5_1'],
                 content_weight: float = 1.0,
                 style_weight: float = 10.0):
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
        
        # Normalize and scale inputs
        generated = F.instance_norm(generated)
        target = F.instance_norm(target)
        
        for name, layer in self.layers.items():
            gen_features = layer(generated)
            target_features = layer(target)
            
            # Normalize features
            gen_features = F.instance_norm(gen_features)
            target_features = F.instance_norm(target_features)
            
            if name in [self.layer_mapping[l] for l in self.content_layers]:
                # Scale content loss
                content_loss += 0.5 * F.mse_loss(gen_features, target_features)
            
            if name in [self.layer_mapping[l] for l in self.style_layers]:
                gen_gram = self.gram_matrix(gen_features)
                target_gram = self.gram_matrix(target_features)
                # Add channel-wise normalization for style
                gen_gram = F.normalize(gen_gram, dim=-1)
                target_gram = F.normalize(target_gram, dim=-1)
                style_loss += F.mse_loss(gen_gram, target_gram)
        
        # Enhanced TV loss
        tv_loss = self.tv_loss(generated)
        
        return {
            'content': content_loss * self.content_weight,
            'style': style_loss * self.style_weight,
            'tv': tv_loss * self.tv_weight
        }

    def tv_loss(self, x):
        """Enhanced TV loss with both L1 and L2 components"""
        diff_i = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        diff_j = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        
        # Combine L1 and L2
        tv_loss = (
            torch.mean(diff_i) + torch.mean(diff_j) +  # L1
            torch.mean(diff_i**2) + torch.mean(diff_j**2)  # L2
        )
        return tv_loss
