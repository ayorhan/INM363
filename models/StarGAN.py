"""
StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation

Source: https://arxiv.org/abs/1711.09020
Type: GAN (Generative Adversarial Network)

Architecture:
- Single generator for multiple domains
- Domain classification discriminator
- Cycle consistency
- Domain conditioning
- Auxiliary classifier

Pros:
- Multi-domain translation
- Efficient single model
- Scalable to many domains
- Shared feature learning

Cons:
- Training complexity increases with domains
- Quality trade-off with domain count
- Domain bias issues
- Limited by shared generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        return x + self.main(x)

class Generator(nn.Module):
    """Generator network for StarGAN."""
    def __init__(self, conv_dim=64, c_dim=5, n_res=6):
        """
        Args:
            conv_dim (int): Number of filters in first conv layer
            c_dim (int): Dimension of domain labels
            n_res (int): Number of residual blocks
        """
        super(Generator, self).__init__()
        self.c_dim = c_dim
        
        # Initial convolution block
        layers = [
            nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling layers
        curr_dim = conv_dim
        for _ in range(2):
            layers.extend([
                nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ])
            curr_dim = curr_dim * 2
        
        # Residual blocks
        for _ in range(n_res):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        
        # Upsampling layers
        for _ in range(2):
            layers.extend([
                nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ])
            curr_dim = curr_dim // 2
        
        # Output layer
        layers.extend([
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        ])
        
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        """
        Args:
            x (Tensor): Input images
            c (Tensor): Target domain labels
            
        Returns:
            Tensor: Generated images
        """
        # Replicate domain labels spatially and concatenate with input
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

class Discriminator(nn.Module):
    """Discriminator network for StarGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        """
        Args:
            image_size (int): Input image size
            conv_dim (int): Number of filters in first conv layer
            c_dim (int): Dimension of domain labels
            repeat_num (int): Number of strided conv layers
        """
        super(Discriminator, self).__init__()
        layers = []
        
        # Initial conv layer
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        
        # Downsampling layers
        curr_dim = conv_dim
        for _ in range(1, repeat_num):
            layers.extend([
                nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.01, inplace=True)
            ])
            curr_dim = curr_dim * 2
        
        # Output layers
        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)  # For real/fake
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)  # For domain classification

    def forward(self, x):
        """
        Args:
            x (Tensor): Input images
            
        Returns:
            tuple: (validity, domain logits)
        """
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), -1)

class StarGAN:
    """StarGAN wrapper class for training and inference."""
    def __init__(self, config):
        """
        Args:
            config (dict): Configuration parameters
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Initialize networks
        self.generator = Generator(
            conv_dim=config['g_conv_dim'],
            c_dim=config['c_dim'],
            n_res=config['n_res']
        ).to(self.device)
        
        self.discriminator = Discriminator(
            image_size=config['image_size'],
            conv_dim=config['d_conv_dim'],
            c_dim=config['c_dim']
        ).to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), 
            lr=config['g_lr'], 
            betas=(config['beta1'], config['beta2'])
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=config['d_lr'], 
            betas=(config['beta1'], config['beta2'])
        )
        
        # Initialize loss weights
        self.lambda_cls = config['lambda_cls']
        self.lambda_rec = config['lambda_rec']
        self.lambda_gp = config['lambda_gp']

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Compute gradient penalty for WGAN-GP."""
        alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates, _ = self.discriminator(interpolates)
        fake = torch.ones(d_interpolates.size()).to(self.device)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_step(self, real_x, real_c, target_c):
        """Perform one training step."""
        # =================================================================================== #
        #                             1. Train the discriminator                               #
        # =================================================================================== #
        
        # Generate fake images
        fake_x = self.generator(real_x, target_c)
        
        # Compute loss with real images
        out_src, out_cls = self.discriminator(real_x)
        d_loss_real = -torch.mean(out_src)
        d_loss_cls = F.cross_entropy(out_cls, real_c)
        
        # Compute loss with fake images
        out_src, _ = self.discriminator(fake_x.detach())
        d_loss_fake = torch.mean(out_src)
        
        # Compute gradient penalty
        gradient_penalty = self.compute_gradient_penalty(real_x, fake_x.detach())
        
        # Backward and optimize
        d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * gradient_penalty
        
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()
        
        # =================================================================================== #
        #                               2. Train the generator                                 #
        # =================================================================================== #
        
        # Original-to-target domain
        fake_x = self.generator(real_x, target_c)
        out_src, out_cls = self.discriminator(fake_x)
        g_loss_fake = -torch.mean(out_src)
        g_loss_cls = F.cross_entropy(out_cls, target_c)
        
        # Target-to-original domain
        reconst_x = self.generator(fake_x, real_c)
        g_loss_rec = torch.mean(torch.abs(real_x - reconst_x))
        
        # Backward and optimize
        g_loss = g_loss_fake + self.lambda_cls * g_loss_cls + self.lambda_rec * g_loss_rec
        
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'd_loss_real': d_loss_real.item(),
            'd_loss_fake': d_loss_fake.item(),
            'd_loss_cls': d_loss_cls.item(),
            'g_loss_cls': g_loss_cls.item(),
            'g_loss_rec': g_loss_rec.item()
        }

    def sample(self, x, c):
        """Generate samples for visualization."""
        with torch.no_grad():
            return self.generator(x.to(self.device), c.to(self.device)) 