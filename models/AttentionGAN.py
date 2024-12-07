"""
AttentionGAN: Attention-Guided Generative Adversarial Networks

Source: https://arxiv.org/abs/1903.12296
Type: GAN (Generative Adversarial Network)

Architecture:
- Attention-guided generator
- Multi-scale discriminator
- Self-attention modules
- Cycle consistency
- Identity preservation

Pros:
- Better handling of geometric changes
- Improved detail preservation
- More precise transformations
- Stable training

Cons:
- Higher computational cost
- Complex architecture
- Memory intensive
- Longer training time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class SelfAttention(nn.Module):
    """
    Self-Attention module for capturing long-range dependencies
    """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = spectral_norm(nn.Conv2d(in_channels, in_channels//8, 1))
        self.key_conv = spectral_norm(nn.Conv2d(in_channels, in_channels//8, 1))
        self.value_conv = spectral_norm(nn.Conv2d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Query, Key, Value projections
        query = self.query_conv(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width*height)
        value = self.value_conv(x).view(batch_size, -1, width*height)
        
        # Attention map
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # Output
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        return self.gamma * out + x

class Generator(nn.Module):
    """
    Generator network with attention mechanisms
    """
    def __init__(self, input_channels=3, output_channels=3, ngf=64):
        super(Generator, self).__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            spectral_norm(nn.Conv2d(input_channels, ngf, 7, padding=3)),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )
        
        # Downsampling
        self.down1 = nn.Sequential(
            spectral_norm(nn.Conv2d(ngf, ngf*2, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            spectral_norm(nn.Conv2d(ngf*2, ngf*4, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(ngf*4),
            nn.ReLU(True)
        )
        
        # Attention blocks
        self.attention1 = SelfAttention(ngf*4)
        self.attention2 = SelfAttention(ngf*4)
        
        # Residual blocks
        self.resblocks = nn.Sequential(*[
            ResidualBlock(ngf*4) for _ in range(9)
        ])
        
        # Upsampling
        self.up1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(ngf*4, ngf*2, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(ngf*2, ngf, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )
        
        # Output convolution
        self.output = nn.Sequential(
            spectral_norm(nn.Conv2d(ngf, output_channels, 7, padding=3)),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Initial convolution and downsampling
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        
        # Apply attention
        x = self.attention1(x)
        
        # Residual blocks
        x = self.resblocks(x)
        
        # Apply attention again
        x = self.attention2(x)
        
        # Upsampling and output
        x = self.up1(x)
        x = self.up2(x)
        x = self.output(x)
        
        return x

class Discriminator(nn.Module):
    """
    Discriminator network with attention mechanisms
    """
    def __init__(self, input_channels=3, ndf=64):
        super(Discriminator, self).__init__()
        
        self.layers = nn.Sequential(
            # Initial convolution
            spectral_norm(nn.Conv2d(input_channels, ndf, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True),
            
            # Downsampling layers
            spectral_norm(nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),
            
            spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),
            
            # Attention layer
            SelfAttention(ndf*4),
            
            spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True),
            
            # Output layer
            spectral_norm(nn.Conv2d(ndf*8, 1, 4, padding=1))
        )
        
    def forward(self, x):
        return self.layers(x)

class ResidualBlock(nn.Module):
    """
    Residual block with instance normalization
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.InstanceNorm2d(channels)
        )
        
    def forward(self, x):
        return x + self.block(x)

class AttentionGAN:
    """
    AttentionGAN model for style transfer
    """
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Initialize networks
        self.G_A = Generator().to(self.device)
        self.G_B = Generator().to(self.device)
        self.D_A = Discriminator().to(self.device)
        self.D_B = Discriminator().to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(
            list(self.G_A.parameters()) + list(self.G_B.parameters()),
            lr=config['lr'],
            betas=(config['beta1'], 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()),
            lr=config['lr'],
            betas=(config['beta1'], 0.999)
        )
        
        # Initialize loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad for networks"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad
                
    def train_step(self, real_A, real_B):
        """
        Perform one training step
        
        Args:
            real_A: Images from domain A
            real_B: Images from domain B
            
        Returns:
            dict: Dictionary containing loss values
        """
        # Forward cycle: A -> B -> A
        fake_B = self.G_A(real_A)
        rec_A = self.G_B(fake_B)
        
        # Backward cycle: B -> A -> B
        fake_A = self.G_B(real_B)
        rec_B = self.G_A(fake_A)
        
        # Identity mapping
        idt_A = self.G_A(real_B)
        idt_B = self.G_B(real_A)
        
        # Train Generators
        self.set_requires_grad([self.D_A, self.D_B], False)
        self.g_optimizer.zero_grad()
        
        # GAN loss
        loss_G_A = self.criterion_GAN(self.D_B(fake_B), torch.ones_like(self.D_B(fake_B)))
        loss_G_B = self.criterion_GAN(self.D_A(fake_A), torch.ones_like(self.D_A(fake_A)))
        loss_G = loss_G_A + loss_G_B
        
        # Cycle consistency loss
        loss_cycle_A = self.criterion_cycle(rec_A, real_A)
        loss_cycle_B = self.criterion_cycle(rec_B, real_B)
        loss_cycle = loss_cycle_A + loss_cycle_B
        
        # Identity loss
        loss_idt_A = self.criterion_identity(idt_A, real_B)
        loss_idt_B = self.criterion_identity(idt_B, real_A)
        loss_idt = loss_idt_A + loss_idt_B
        
        # Total generator loss
        loss_G_total = (loss_G + 
                       self.config['lambda_cycle'] * loss_cycle +
                       self.config['lambda_identity'] * loss_idt)
        
        loss_G_total.backward()
        self.g_optimizer.step()
        
        # Train Discriminators
        self.set_requires_grad([self.D_A, self.D_B], True)
        self.d_optimizer.zero_grad()
        
        # Discriminator A
        loss_D_A_real = self.criterion_GAN(
            self.D_A(real_A), torch.ones_like(self.D_A(real_A)))
        loss_D_A_fake = self.criterion_GAN(
            self.D_A(fake_A.detach()), torch.zeros_like(self.D_A(fake_A.detach())))
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
        
        # Discriminator B
        loss_D_B_real = self.criterion_GAN(
            self.D_B(real_B), torch.ones_like(self.D_B(real_B)))
        loss_D_B_fake = self.criterion_GAN(
            self.D_B(fake_B.detach()), torch.zeros_like(self.D_B(fake_B.detach())))
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
        
        # Total discriminator loss
        loss_D_total = loss_D_A + loss_D_B
        loss_D_total.backward()
        self.d_optimizer.step()
        
        return {
            'G_loss': loss_G_total.item(),
            'D_loss': loss_D_total.item(),
            'G_adv': loss_G.item(),
            'cycle_loss': loss_cycle.item(),
            'idt_loss': loss_idt.item()
        }
    
    def transfer_style(self, content_image, style_image, direction='AtoB'):
        """
        Transfer style between domains
        
        Args:
            content_image: Content image tensor
            style_image: Style image tensor
            direction: Transfer direction ('AtoB' or 'BtoA')
            
        Returns:
            Tensor: Stylized image
        """
        self.G_A.eval()
        self.G_B.eval()
        
        with torch.no_grad():
            if direction == 'AtoB':
                return self.G_A(content_image)
            else:
                return self.G_B(content_image) 