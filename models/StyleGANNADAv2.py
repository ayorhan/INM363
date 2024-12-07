"""
StyleGAN-NADA v2: Enhanced Text-Driven Domain Adaptation

Source: Extension of StyleGAN-NADA (https://arxiv.org/abs/2108.00946)
Type: GAN (Generative Adversarial Network)

Architecture:
- Enhanced StyleGAN2 backbone
- Improved CLIP guidance
- Multi-scale discriminator
- Adaptive feature modulation
- Enhanced text conditioning

Pros:
- Better text-driven control
- Improved stability
- Higher quality outputs
- More flexible adaptation

Cons:
- Very computationally intensive
- Large model size
- Complex training process
- High memory requirements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from collections import OrderedDict
import numpy as np

class EqualLinear(nn.Module):
    """Linear layer with equalized learning rate."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.scale = (in_dim) ** -0.5

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)

class CLIPTextEncoder(nn.Module):
    """CLIP-based text encoder wrapper"""
    def __init__(self, c_dim):
        super().__init__()
        self.model, _ = clip.load("ViT-B/32")
        self.model.eval()
        self.c_dim = c_dim
        self.projection = nn.Linear(512, c_dim)  # CLIP's default output is 512
        
    def forward(self, text):
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        return self.projection(text_features)

class StyleGANGenerator(nn.Module):
    """Enhanced StyleGAN generator with improved domain adaptation"""
    def __init__(self, z_dim=512, w_dim=512, c_dim=512, img_resolution=1024, img_channels=3):
        super(StyleGANGenerator, self).__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        
        # Enhanced mapping network
        self.mapping = EnhancedMappingNetwork(
            z_dim=z_dim,
            w_dim=w_dim,
            c_dim=c_dim,
            num_layers=8
        )
        
        # Improved synthesis network
        self.synthesis = EnhancedSynthesisNetwork(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels
        )
        
        # Domain adaptation module
        self.domain_adapter = DomainAdaptationModule(w_dim)
        
    def forward(self, z, c=None, truncation_psi=1.0):
        w = self.mapping(z, c)
        w = self.domain_adapter(w)
        img = self.synthesis(w, truncation_psi)
        return img

class EnhancedMappingNetwork(nn.Module):
    """Enhanced mapping network with improved text conditioning"""
    def __init__(self, z_dim, w_dim, c_dim, num_layers):
        super(EnhancedMappingNetwork, self).__init__()
        
        # Text encoder (CLIP-based)
        self.text_encoder = CLIPTextEncoder(c_dim)
        
        # Main mapping layers
        layers = []
        dim = z_dim
        for i in range(num_layers):
            layers.append(
                ('dense%d' % i, EqualLinear(dim, w_dim if i == num_layers-1 else dim))
            )
            if i != num_layers-1:
                layers.append(('act%d' % i, nn.LeakyReLU(0.2)))
        self.main = nn.Sequential(OrderedDict(layers))
        
        # Style mixing module
        self.style_mixing = StyleMixingModule(w_dim)
        
    def forward(self, z, c=None):
        # Text conditioning
        if c is not None:
            text_features = self.text_encoder(c)
            z = torch.cat([z, text_features], dim=1)
            
        # Main mapping
        w = self.main(z)
        
        # Style mixing
        w = self.style_mixing(w)
        return w

class EnhancedSynthesisNetwork(nn.Module):
    """Improved synthesis network with enhanced feature modulation"""
    def __init__(self, w_dim, img_resolution, img_channels):
        super(EnhancedSynthesisNetwork, self).__init__()
        
        self.img_resolution = img_resolution
        self.w_dim = w_dim
        
        # Number of synthesis blocks
        self.num_layers = int(np.log2(img_resolution)) * 2 - 2
        
        # Synthesis blocks
        self.blocks = nn.ModuleList()
        in_channels = w_dim
        out_channels = w_dim
        
        for i in range(self.num_layers):
            if i % 2 == 0:
                out_channels = in_channels * 2
            self.blocks.append(
                SynthesisBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    w_dim=w_dim,
                    resolution=2 ** (i//2 + 2)
                )
            )
            in_channels = out_channels
            
        # ToRGB layers
        self.to_rgb = nn.ModuleList([
            ToRGBLayer(out_channels, img_channels, w_dim)
            for _ in range(len(self.blocks))
        ])
        
    def forward(self, w, truncation_psi=1.0):
        # Split w into layers
        w = w.unsqueeze(1).repeat(1, self.num_layers, 1)
        
        # Apply truncation
        if truncation_psi != 1.0:
            w_avg = self.mean_latent(truncation_psi)
            w = w_avg + truncation_psi * (w - w_avg)
            
        # Generate image
        x = self.blocks[0](w[:, 0])
        rgb = self.to_rgb[0](x, w[:, 0])
        
        for i in range(1, len(self.blocks)):
            x = self.blocks[i](x, w[:, i])
            rgb_new = self.to_rgb[i](x, w[:, i])
            rgb = F.interpolate(rgb, scale_factor=2) + rgb_new
            
        return rgb

class DomainAdaptationModule(nn.Module):
    """Enhanced domain adaptation module"""
    def __init__(self, w_dim):
        super(DomainAdaptationModule, self).__init__()
        
        self.mapper = nn.Sequential(
            nn.Linear(w_dim, w_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(w_dim, w_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(w_dim, w_dim)
        )
        
        self.attention = SelfAttention(w_dim)
        
    def forward(self, w):
        w_mapped = self.mapper(w)
        w_attended = self.attention(w_mapped)
        return w + w_attended

class StyleGANNADAv2(nn.Module):
    """Main StyleGAN-NADA v2 model"""
    def __init__(self, config):
        super(StyleGANNADAv2, self).__init__()
        self.device = config['device']
        self.config = config
        
        # Initialize networks
        self.generator = StyleGANGenerator(
            z_dim=config['z_dim'],
            w_dim=config['w_dim'],
            c_dim=config['c_dim']
        ).to(self.device)
        
        # Enhanced CLIP model
        self.clip_model = EnhancedCLIPModel().to(self.device)
        
        # Multi-scale discriminator
        self.discriminator = MultiScaleDiscriminator().to(self.device)
        
        # Initialize optimizers
        self.setup_optimizers()
        
    def setup_optimizers(self):
        self.g_opt = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config['g_lr'],
            betas=(0.9, 0.999)
        )
        self.d_opt = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config['d_lr'],
            betas=(0.9, 0.999)
        )
        
    def compute_clip_loss(self, images, text_prompt):
        """Enhanced CLIP-based loss computation"""
        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(text_prompt)
        
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity with temperature scaling
        similarity = torch.matmul(image_features, text_features.T) / self.config['temperature']
        return -similarity.mean()
    
    def train_step(self, z, text_prompt, source_prompt):
        """Enhanced training step"""
        self.g_opt.zero_grad()
        self.d_opt.zero_grad()
        
        # Generate images
        generated = self.generator(z)
        
        # Multi-scale discriminator loss
        d_real, d_fake = self.discriminator(generated)
        d_loss = self.compute_adversarial_loss(d_real, d_fake)
        
        # Enhanced CLIP directional loss
        clip_loss = self.compute_clip_loss(generated, text_prompt)
        
        # Identity preservation with source prompt
        id_loss = self.compute_clip_loss(generated, source_prompt)
        
        # Total loss with adaptive weighting
        total_loss = (
            self.config['adv_weight'] * d_loss +
            self.config['clip_weight'] * clip_loss +
            self.config['id_weight'] * id_loss
        )
        
        # Backward pass and optimization
        total_loss.backward()
        self.g_opt.step()
        self.d_opt.step()
        
        return {
            'd_loss': d_loss.item(),
            'clip_loss': clip_loss.item(),
            'id_loss': id_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def generate_images(self, z, text_prompt):
        """Generate images with enhanced control"""
        self.generator.eval()
        with torch.no_grad():
            # Get text conditioning
            text_features = self.clip_model.encode_text(text_prompt)
            # Generate images
            return self.generator(z, text_features)

class SelfAttention(nn.Module):
    """Self-attention module for enhanced feature processing"""
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        return attn @ v

class SynthesisBlock(nn.Module):
    """Basic synthesis block for StyleGAN"""
    def __init__(self, w_dim, layer_idx):
        super().__init__()
        self.conv1 = nn.Conv2d(w_dim, w_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(w_dim, w_dim, 3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear') if layer_idx > 0 else nn.Identity()

    def forward(self, x, w=None):
        x = self.upsample(x)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return x

class ToRGBLayer(nn.Module):
    """Convert features to RGB output"""
    def __init__(self, w_dim, img_channels):
        super().__init__()
        self.conv = nn.Conv2d(w_dim, img_channels, 1)
        
    def forward(self, x, w=None):
        return self.conv(x)

class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for improved stability"""
    def __init__(self, channels=3, base_features=64, n_scales=3):
        super().__init__()
        self.discriminators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, base_features * (2**i), 4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(base_features * (2**i), base_features * (2**(i+1)), 4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(base_features * (2**(i+1)), 1, 1)
            ) for i in range(n_scales)
        ])
        
    def forward(self, x):
        return torch.cat([disc(x) for disc in self.discriminators], dim=1).mean()

class StyleMixingModule(nn.Module):
    """Module for style mixing regularization"""
    def __init__(self, w_dim):
        super().__init__()
        self.w_dim = w_dim
        
    def forward(self, w):
        batch_size = w.size(0)
        if self.training and batch_size > 1:
            # Randomly mix styles with probability 0.9
            if torch.rand(1).item() < 0.9:
                cutoff = torch.randint(1, w.size(1), (1,)).item()
                w2 = w[torch.randperm(batch_size)]
                w[:, cutoff:] = w2[:, cutoff:]
        return w

class EnhancedCLIPModel(nn.Module):
    """Enhanced CLIP model wrapper with additional functionality"""
    def __init__(self):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
    def encode_image(self, images):
        return self.model.encode_image(images)
        
    def encode_text(self, text):
        return self.model.encode_text(text)