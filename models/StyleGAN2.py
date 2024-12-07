"""
StyleGAN2: Analyzing and Improving the Image Quality of StyleGAN

Source: https://arxiv.org/abs/1912.04958
Type: GAN (Generative Adversarial Network)

Architecture:
- Improved mapping network
- Path length regularization
- Modulated convolutions
- Weight demodulation
- No progressive growing

Pros:
- High quality generation
- Better training stability
- Reduced artifacts
- Style mixing capability

Cons:
- Extensive compute requirements
- Large model size
- Long training time
- High memory usage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MappingNetwork(nn.Module):
    """
    Mapping network to transform input latent vectors into intermediate W space
    """
    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(z_dim if i == 0 else w_dim, w_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        self.mapping = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                m.bias.data.zero_()
    
    def forward(self, z):
        """
        Args:
            z (Tensor): Input noise vector [B, z_dim]
        Returns:
            Tensor: Transformed style vector w [B, w_dim]
        """
        # Normalize input
        z = F.normalize(z, dim=1)
        # Map to W space
        w = self.mapping(z)
        return w

class ModulatedConv2d(nn.Module):
    """
    Modulated Convolution layer with style modulation and demodulation
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 style_dim=512, demodulate=True, upsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        
        # Conv weight
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        
        # Style modulation
        self.modulation = nn.Linear(style_dim, in_channels)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.weight, a=0.2)
        nn.init.kaiming_normal_(self.modulation.weight, a=0.2)
        self.modulation.bias.data.zero_()
    
    def forward(self, x, style):
        batch, in_channels, height, width = x.shape
        
        # Style modulation
        style = self.modulation(style).view(batch, 1, in_channels, 1, 1)
        weight = self.weight.unsqueeze(0) * style
        
        # Demodulation
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)
        
        # Reshape for batch multiplication
        weight = weight.view(
            batch * self.out_channels, in_channels, 
            self.kernel_size, self.kernel_size
        )
        
        # Reshape input
        x = x.view(1, batch * in_channels, height, width)
        
        # Convolution
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            padding = (self.kernel_size - 1) // 2
            out = F.conv2d(x, weight, padding=padding, groups=batch)
        else:
            padding = (self.kernel_size - 1) // 2
            out = F.conv2d(x, weight, padding=padding, groups=batch)
        
        # Reshape output
        out = out.view(batch, self.out_channels, *out.shape[2:])
        
        return out + self.bias

class StyleConvBlock(nn.Module):
    """
    StyleGAN2 Convolutional Block with style modulation
    """
    def __init__(self, in_channels, out_channels, style_dim, 
                 upsample=False, noise=True):
        super().__init__()
        self.conv1 = ModulatedConv2d(
            in_channels, out_channels, 3, 
            style_dim=style_dim, upsample=upsample
        )
        self.noise1 = NoiseInjection() if noise else None
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x, style, noise=None):
        out = self.conv1(x, style)
        if self.noise1 is not None:
            out = self.noise1(out, noise)
        out = self.activate(out)
        return out

class NoiseInjection(nn.Module):
    """
    Add noise to feature maps
    """
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, noise=None):
        if noise is None:
            batch, _, height, width = x.shape
            noise = torch.randn(batch, 1, height, width).to(x.device)
        return x + self.weight * noise

class ToRGB(nn.Module):
    """
    Convert features to RGB
    """
    def __init__(self, in_channels, style_dim):
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channels, 3, 1, 
            style_dim=style_dim, demodulate=False
        )
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
    
    def forward(self, x, style):
        out = self.conv(x, style)
        return out + self.bias

class StyleGAN2Generator(nn.Module):
    """
    StyleGAN2 Generator with improved architecture
    """
    def __init__(self, z_dim=512, w_dim=512, num_mapping=8, 
                 num_layers=18, start_channels=32):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_layers = num_layers
        
        # Mapping network
        self.mapping = MappingNetwork(z_dim, w_dim, num_mapping)
        
        # Initial constant input
        self.input = nn.Parameter(torch.randn(1, start_channels, 4, 4))
        
        # Style convolution blocks
        self.style_blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        
        in_channels = start_channels
        for i in range(num_layers):
            if i % 2 == 0 and i > 0:
                out_channels = in_channels // 2
            else:
                out_channels = in_channels
                
            self.style_blocks.append(
                StyleConvBlock(
                    in_channels, out_channels, w_dim,
                    upsample=(i % 2 == 0 and i > 0)
                )
            )
            
            if i % 2 == 0 or i == num_layers - 1:
                self.to_rgb.append(ToRGB(out_channels, w_dim))
            
            in_channels = out_channels
    
    def forward(self, z, truncation=1.0, truncation_latent=None):
        """
        Args:
            z (Tensor): Input noise vector [B, z_dim]
            truncation (float): Truncation factor for style mixing
            truncation_latent (Tensor, optional): Mean w for truncation
        """
        # Map z to w space
        w = self.mapping(z)
        
        # Apply truncation trick
        if truncation < 1:
            if truncation_latent is None:
                truncation_latent = torch.zeros_like(w).mean(0, keepdim=True)
            w = truncation_latent + truncation * (w - truncation_latent)
        
        # Initial input
        batch = w.shape[0]
        x = self.input.repeat(batch, 1, 1, 1)
        
        # Generate image
        rgb = None
        for i, (block, to_rgb) in enumerate(zip(self.style_blocks, self.to_rgb)):
            x = block(x, w)
            
            if i % 2 == 0 or i == len(self.style_blocks) - 1:
                if rgb is not None:
                    rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)
                new_rgb = to_rgb(x, w)
                rgb = new_rgb if rgb is None else rgb + new_rgb
        
        return rgb

class StyleGAN2Discriminator(nn.Module):
    """
    StyleGAN2 Discriminator with residual architecture
    """
    def __init__(self, input_size=1024, start_channels=32):
        super().__init__()
        
        # Initial convolution
        self.from_rgb = nn.Sequential(
            nn.Conv2d(3, start_channels, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Discriminator blocks
        self.blocks = nn.ModuleList()
        in_channels = start_channels
        
        for i in range(int(math.log2(input_size)) - 2):
            out_channels = min(in_channels * 2, 512)
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = out_channels
        
        # Final layers
        self.final = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, 1, 4, 1, 0)
        )
    
    def forward(self, x):
        # Initial RGB conversion
        x = self.from_rgb(x)
        
        # Feature extraction
        for block in self.blocks:
            x = block(x)
        
        # Final prediction
        return self.final(x).squeeze()

class StyleGAN2:
    """
    StyleGAN2 wrapper class for training and inference
    """
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Initialize networks
        self.generator = StyleGAN2Generator(
            z_dim=config['z_dim'],
            w_dim=config['w_dim'],
            num_mapping=config['num_mapping'],
            num_layers=config['num_layers']
        ).to(self.device)
        
        self.discriminator = StyleGAN2Discriminator(
            input_size=config['image_size'],
            start_channels=config['start_channels']
        ).to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=config['g_lr'],
            betas=(0.0, 0.99)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config['d_lr'],
            betas=(0.0, 0.99)
        )
        
        # Initialize path length regularization
        self.path_batch_shrink = 2
        self.path_lengths = torch.zeros(500).to(self.device)
        self.path_length_idx = 0
        
    def path_length_penalty(self, w, output):
        """Calculate path length penalty for generator regularization"""
        device = w.device
        noise = torch.randn_like(output) / math.sqrt(output.shape[2] * output.shape[3])
        grad, = torch.autograd.grad(
            outputs=(output * noise).sum(),
            inputs=w,
            create_graph=True
        )
        
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
        path_mean = self.path_lengths.mean()
        
        path_penalty = (path_lengths - path_mean).pow(2).mean()
        
        # Update path length mean
        self.path_lengths[self.path_length_idx] = path_lengths.mean().detach()
        self.path_length_idx = (self.path_length_idx + 1) % 500
        
        return path_penalty

    def train_step(self, real_images):
        """Perform one training step"""
        batch_size = real_images.size(0)
        device = real_images.device
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        z = torch.randn(batch_size, self.config['z_dim']).to(device)
        with torch.no_grad():
            fake_images = self.generator(z)
            
        real_pred = self.discriminator(real_images)
        fake_pred = self.discriminator(fake_images)
        
        d_loss = F.softplus(fake_pred).mean() + F.softplus(-real_pred).mean()
        
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        z = torch.randn(batch_size, self.config['z_dim']).to(device)
        fake_images = self.generator(z)
        fake_pred = self.discriminator(fake_images)
        
        g_loss = F.softplus(-fake_pred).mean()
        
        # Path length regularization
        if self.config['path_reg_weight'] > 0:
            batch_size = max(1, batch_size // self.path_batch_shrink)
            z = torch.randn(batch_size, self.config['z_dim']).to(device)
            w = self.generator.mapping(z)
            fake_images = self.generator(z, w)
            
            path_loss = self.path_length_penalty(w, fake_images)
            g_loss = g_loss + self.config['path_reg_weight'] * path_loss
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'real_score': real_pred.mean().item(),
            'fake_score': fake_pred.mean().item()
        }

    def generate(self, num_samples=1, truncation=0.7):
        """Generate samples"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.config['z_dim']).to(self.device)
            samples = self.generator(z, truncation=truncation)
        return samples
