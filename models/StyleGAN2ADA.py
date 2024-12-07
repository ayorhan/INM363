"""
StyleGAN2-ADA: Training Generative Adversarial Networks with Limited Data

Source: https://arxiv.org/abs/2006.06676
Type: GAN (Generative Adversarial Network)

Architecture:
- Adaptive discriminator augmentation
- StyleGAN2 backbone
- Path length regularization
- Adaptive feature normalization
- Progressive growing

Pros:
- Works with limited data
- Improved training stability
- Better quality with small datasets
- Reduced overfitting

Cons:
- Complex implementation
- High computational requirements
- Sensitive augmentation pipeline
- Training instability risks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_utils import persistence
from torch_utils import misc
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma

@persistence.persistent_class
class MappingNetwork(nn.Module):
    def __init__(self, z_dim, c_dim, w_dim, num_ws):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        
        # Simple mapping network - can be expanded for better performance
        self.net = nn.Sequential(
            nn.Linear(z_dim, w_dim),
            nn.ReLU(),
            nn.Linear(w_dim, w_dim * num_ws)
        )
    
    def forward(self, z, c=None, truncation_psi=1, truncation_cutoff=None):
        w = self.net(z)
        w = w.view(-1, self.num_ws, self.w_dim)
        return w

@persistence.persistent_class
class SynthesisNetwork(nn.Module):
    def __init__(self, w_dim, img_resolution, img_channels):
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_ws = int(np.log2(img_resolution)) * 2 - 2
        
        # Basic synthesis network - you may want to expand this
        self.conv = nn.ModuleList([
            nn.Conv2d(w_dim, img_channels, 3, padding=1)
            for _ in range(self.num_ws)
        ])
    
    def forward(self, w):
        x = w.new_zeros(w.shape[0], self.img_channels, self.img_resolution, self.img_resolution)
        for i, conv in enumerate(self.conv):
            x = conv(x)
        return x

class StyleGAN2ADA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config['device']
        
        # Initialize networks
        self.G = Generator(
            z_dim=config['z_dim'],
            w_dim=config['w_dim'],
            c_dim=config['c_dim'],
            img_resolution=config['img_resolution'],
            img_channels=config['img_channels']
        ).to(self.device)
        
        self.D = Discriminator(
            c_dim=config['c_dim'],
            img_resolution=config['img_resolution'],
            img_channels=config['img_channels']
        ).to(self.device)
        
        # Augmentation pipeline
        self.augment_pipe = AugmentPipe(
            xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1,
            brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1
        ).to(self.device)
        
        # Initialize optimizers
        self.G_opt = torch.optim.Adam(
            self.G.parameters(),
            lr=config['g_lr'],
            betas=(0.0, 0.99)
        )
        self.D_opt = torch.optim.Adam(
            self.D.parameters(),
            lr=config['d_lr'],
            betas=(0.0, 0.99)
        )

    def generate_mixing_regularization(self, batch_size, mixing_prob):
        """Generate latents with style mixing regularization"""
        z = torch.randn(batch_size, self.config['z_dim']).to(self.device)
        if torch.rand(()).item() < mixing_prob:
            z2 = torch.randn(batch_size, self.config['z_dim']).to(self.device)
            mixing_cutoff = torch.randint(1, self.G.num_ws, ()).item()
            z = torch.cat([z[:, :mixing_cutoff], z2[:, mixing_cutoff:]], dim=1)
        return z

    def train_step(self, real_img, z=None):
        """Perform one training step"""
        batch_size = real_img.size(0)
        
        # Generate latents
        if z is None:
            z = self.generate_mixing_regularization(
                batch_size, 
                mixing_prob=self.config['mixing_prob']
            )
        
        # Update discriminator
        self.D_opt.zero_grad()
        
        # Real images with augmentation
        real_img_aug = self.augment_pipe(real_img)
        real_logits = self.D(real_img_aug)
        
        # Generate fake images
        with torch.no_grad():
            fake_img = self.G(z)
        fake_img_aug = self.augment_pipe(fake_img)
        fake_logits = self.D(fake_img_aug)
        
        # Compute D loss
        d_loss = self.compute_d_loss(real_logits, fake_logits)
        d_loss.backward()
        self.D_opt.step()
        
        # Update generator
        self.G_opt.zero_grad()
        
        fake_img = self.G(z)
        fake_img_aug = self.augment_pipe(fake_img)
        fake_logits = self.D(fake_img_aug)
        
        # Compute G loss
        g_loss = self.compute_g_loss(fake_logits)
        g_loss.backward()
        self.G_opt.step()
        
        # Update augmentation probability
        self.augment_pipe.update()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'augment_p': self.augment_pipe.p.item()
        }

class AugmentPipe(nn.Module):
    """Augmentation pipeline with adaptive probability"""
    def __init__(self, **kwargs):
        super().__init__()
        self.p = nn.Parameter(torch.zeros([]))
        self.augment_specs = {k: v for k, v in kwargs.items() if v > 0}
        
    def forward(self, images):
        if self.p.item() <= 0 or not self.training:
            return images
            
        # Apply augmentations
        for name, strength in self.augment_specs.items():
            if torch.rand(()).item() < self.p.item():
                images = self._apply_augment(images, name, strength)
                
        return images
    
    def _apply_augment(self, images, name, strength):
        if name == 'xflip':
            return torch.flip(images, [3])
        elif name == 'rotate90':
            return torch.rot90(images, k=torch.randint(1, 4, ()).item(), dims=[2, 3])
        # ... implement other augmentations ...
        return images
        
    def update(self):
        # Update augmentation probability based on overfitting heuristic
        # This is a simplified version
        self.p.data = torch.clamp(self.p + 0.01, 0, 1)

@persistence.persistent_class
class Generator(nn.Module):
    def __init__(self, z_dim, w_dim, c_dim, img_resolution, img_channels):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels
        )
        self.mapping = MappingNetwork(
            z_dim=z_dim,
            c_dim=c_dim,
            w_dim=w_dim,
            num_ws=self.synthesis.num_ws
        )
        
    def forward(self, z, c=None, truncation_psi=1, truncation_cutoff=None):
        w = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(w)
        return img

@persistence.persistent_class
class Discriminator(nn.Module):
    def __init__(self, c_dim, img_resolution, img_channels):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        
        # Main layers
        self.main = nn.Sequential(
            # Initial convolution
            nn.Conv2d(img_channels, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            
            # Downsampling blocks
            *[DiscriminatorBlock(
                in_channels=min(64 * 2**i, 512),
                out_channels=min(64 * 2**(i+1), 512)
            ) for i in range(int(np.log2(img_resolution)) - 2)],
            
            # Final layers
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1 + c_dim, 4, stride=1, padding=0)
        )
        
    def forward(self, img, c=None):
        x = self.main(img)
        out = x.squeeze(2).squeeze(2)
        if c is not None:
            out = out[:, :1]  # Keep only prediction, discard class outputs
        return out

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=2, padding=0)
        
    def forward(self, x):
        skip = self.skip(x)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)
        return (x + skip) / np.sqrt(2) 