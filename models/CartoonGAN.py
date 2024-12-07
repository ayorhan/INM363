"""
CartoonGAN: Generative Adversarial Networks for Photo Cartoonization

Source: https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018.pdf
Type: GAN (Generative Adversarial Network)

Architecture:
- ResNet-based generator with instance normalization
- Multi-scale discriminator with spectral normalization
- Edge-promoting adversarial loss
- Semantic consistency preservation
- Two-stage training process

Pros:
- High-quality cartoon style transfer
- Preserves content structure
- Consistent style across images
- Handles diverse input photos

Cons:
- Style limited to trained cartoon domain
- Can produce artifacts on complex scenes
- Requires large cartoon dataset
- Computationally intensive training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class ResidualBlock(nn.Module):
    """
    Residual block for generator network
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    """
    Generator network for CartoonGAN
    Transforms photos into cartoon-style images
    """
    def __init__(self, input_channels=3, output_channels=3, num_filters=64, num_blocks=8):
        super(Generator, self).__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, 7, padding=3),
            nn.InstanceNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling layers
        self.down1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(num_filters*2),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(num_filters*2, num_filters*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(num_filters*4),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters*4) for _ in range(num_blocks)]
        )
        
        # Upsampling layers
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(num_filters*4, num_filters*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(num_filters*2),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(num_filters*2, num_filters, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        
        # Output convolution
        self.output = nn.Sequential(
            nn.Conv2d(num_filters, output_channels, 7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.output(x)
        return x

class Discriminator(nn.Module):
    """
    Discriminator network for CartoonGAN
    Distinguishes between real cartoons and generated images
    """
    def __init__(self, input_channels=3, num_filters=64):
        super(Discriminator, self).__init__()
        
        self.layers = nn.Sequential(
            # Initial convolution
            nn.Conv2d(input_channels, num_filters, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsampling layers
            nn.Conv2d(num_filters, num_filters*2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters*2, num_filters*4, 3, stride=1, padding=1),
            nn.InstanceNorm2d(num_filters*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(num_filters*4, num_filters*8, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters*8, num_filters*8, 3, stride=1, padding=1),
            nn.InstanceNorm2d(num_filters*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer
            nn.Conv2d(num_filters*8, 1, 3, stride=1, padding=1)
        )
        
    def forward(self, x):
        return self.layers(x)

class EdgePromotingLoss(nn.Module):
    """
    Edge-promoting loss for cartoon-style transfer
    Encourages clear edges in generated images
    """
    def __init__(self):
        super(EdgePromotingLoss, self).__init__()
        self.edge_filter = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        
        # Sobel filter for edge detection
        sobel_kernel = torch.tensor([[-1, -2, -1],
                                   [ 0,  0,  0],
                                   [ 1,  2,  1]], dtype=torch.float32)
        sobel_kernel = sobel_kernel.view(1, 1, 3, 3)
        sobel_kernel = sobel_kernel.repeat(3, 1, 1, 1)
        self.edge_filter.weight.data = sobel_kernel
        self.edge_filter.weight.requires_grad = False
        
    def forward(self, y_true, y_pred):
        true_edges = self.edge_filter(y_true)
        pred_edges = self.edge_filter(y_pred)
        return F.l1_loss(pred_edges, true_edges)

class CartoonGAN:
    """
    CartoonGAN model for cartoon style transfer
    """
    def __init__(self, config):
        self.device = config['device']
        self.config = config
        
        # Initialize networks
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # Initialize VGG for feature extraction
        vgg = models.vgg19(pretrained=True).features[:20].eval().to(self.device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        
        # Initialize edge-promoting loss
        self.edge_loss = EdgePromotingLoss().to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=config['lr'],
            betas=(config['beta1'], 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config['lr'],
            betas=(config['beta1'], 0.999)
        )
        
    def initialize_with_pretrain(self, num_epochs=10):
        """Initialize generator with reconstruction loss"""
        self.generator.train()
        
        for epoch in range(num_epochs):
            for batch in self.config['dataloader']:
                photos = batch['photo'].to(self.device)
                
                # Generate images
                generated = self.generator(photos)
                
                # Compute reconstruction loss
                recon_loss = F.l1_loss(generated, photos)
                
                # Update generator
                self.g_optimizer.zero_grad()
                recon_loss.backward()
                self.g_optimizer.step()
                
    def train_step(self, photo_batch, cartoon_batch, smooth_cartoon_batch):
        """
        Perform one training step
        
        Args:
            photo_batch: Batch of photo images
            cartoon_batch: Batch of cartoon images
            smooth_cartoon_batch: Batch of smoothed cartoon images
            
        Returns:
            dict: Dictionary containing loss values
        """
        # Move data to device
        real_photos = photo_batch.to(self.device)
        real_cartoons = cartoon_batch.to(self.device)
        smooth_cartoons = smooth_cartoon_batch.to(self.device)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Generate fake cartoons
        fake_cartoons = self.generator(real_photos)
        
        # Real cartoon discriminator loss
        real_validity = self.discriminator(real_cartoons)
        d_real_loss = torch.mean(torch.square(real_validity - 1))
        
        # Fake cartoon discriminator loss
        fake_validity = self.discriminator(fake_cartoons.detach())
        d_fake_loss = torch.mean(torch.square(fake_validity))
        
        # Smooth cartoon discriminator loss
        smooth_validity = self.discriminator(smooth_cartoons)
        d_smooth_loss = torch.mean(torch.square(smooth_validity))
        
        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss + d_smooth_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        # Adversarial loss
        fake_validity = self.discriminator(fake_cartoons)
        g_adv_loss = torch.mean(torch.square(fake_validity - 1))
        
        # Content loss
        real_features = self.vgg(real_photos)
        fake_features = self.vgg(fake_cartoons)
        content_loss = F.l1_loss(fake_features, real_features)
        
        # Edge-promoting loss
        edge_loss = self.edge_loss(real_cartoons, fake_cartoons)
        
        # Total generator loss
        g_loss = (self.config['lambda_adv'] * g_adv_loss +
                 self.config['lambda_content'] * content_loss +
                 self.config['lambda_edge'] * edge_loss)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'content_loss': content_loss.item(),
            'edge_loss': edge_loss.item()
        }
    
    def cartoonize(self, photo):
        """
        Transform a photo into cartoon style
        
        Args:
            photo: Input photo tensor
            
        Returns:
            Tensor: Cartoonized image
        """
        self.generator.eval()
        with torch.no_grad():
            return self.generator(photo.to(self.device))
    
    def save_models(self, path):
        """Save model checkpoints"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
        }, path)
    
    def load_models(self, path):
        """Load model checkpoints"""
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict']) 