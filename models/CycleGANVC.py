"""
CycleGAN-VC: Voice Conversion with CycleGAN

Source: https://arxiv.org/abs/1904.04631
Type: GAN (Generative Adversarial Network)

Architecture:
- 2D CNN generators
- PatchGAN discriminators
- Identity-preserving module
- Cycle consistency
- Mel-spectrogram processing

Pros:
- Non-parallel voice conversion
- Identity preservation
- Good audio quality
- Stable training

Cons:
- Limited by cycle consistency
- Audio artifacts possible
- Speaker-specific training
- Resource intensive
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GLU(nn.Module):
    """
    Gated Linear Unit for voice conversion
    """
    def __init__(self):
        super(GLU, self).__init__()
        
    def forward(self, x):
        return x * torch.sigmoid(x)

class DownSampleBlock(nn.Module):
    """
    Downsample block for generator
    Uses 2D convolution for processing mel-spectrograms
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DownSampleBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_channels),
            GLU()
        )
        
    def forward(self, x):
        return self.conv(x)

class UpSampleBlock(nn.Module):
    """
    Upsample block for generator
    Uses transposed convolution for upsampling mel-spectrograms
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UpSampleBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_channels),
            GLU()
        )
        
    def forward(self, x):
        return self.conv(x)

class Generator(nn.Module):
    """
    Generator network for voice conversion
    Converts mel-spectrograms between domains
    """
    def __init__(self):
        super(Generator, self).__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(5, 15), padding=(2, 7)),
            nn.InstanceNorm2d(128),
            GLU()
        )
        
        # Downsampling layers
        self.down_blocks = nn.ModuleList([
            DownSampleBlock(64, 256, (4, 4), (2, 2), (1, 1)),
            DownSampleBlock(128, 512, (4, 4), (2, 2), (1, 1))
        ])
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(6)
        ])
        
        # Upsampling layers
        self.up_blocks = nn.ModuleList([
            UpSampleBlock(256, 256, (4, 4), (2, 2), (1, 1)),
            UpSampleBlock(128, 128, (4, 4), (2, 2), (1, 1))
        ])
        
        # Output convolution
        self.output = nn.Conv2d(64, 1, kernel_size=(5, 15), padding=(2, 7))
        
    def forward(self, x):
        # Initial convolution
        x = self.initial(x)
        
        # Downsampling
        skip_connections = []
        for down_block in self.down_blocks:
            skip_connections.append(x)
            x = down_block(x)
            
        # Residual blocks
        for res_block in self.residual_blocks:
            x = res_block(x)
            
        # Upsampling with skip connections
        for up_block, skip in zip(self.up_blocks, reversed(skip_connections)):
            x = up_block(x)
            x = torch.cat((x, skip), dim=1)
            
        # Output
        return self.output(x)

class Discriminator(nn.Module):
    """
    Discriminator network for voice conversion
    Uses 2D convolutions for processing mel-spectrograms
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.layers = nn.Sequential(
            # Initial convolution
            nn.Conv2d(1, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(0.2, True),
            
            # Downsampling layers
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.InstanceNorm2d(1024),
            nn.LeakyReLU(0.2, True),
            
            # Output layer
            nn.Conv2d(1024, 1, kernel_size=(1, 1))
        )
        
    def forward(self, x):
        return self.layers(x)

class ResidualBlock(nn.Module):
    """
    Residual block for generator
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.InstanceNorm2d(channels),
            GLU(),
            nn.Conv2d(channels//2, channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.InstanceNorm2d(channels)
        )
        
    def forward(self, x):
        return x + self.block(x)

class CycleGANVC:
    """
    CycleGAN-VC model for voice conversion
    """
    def __init__(self, config):
        self.device = config['device']
        self.config = config
        
        # Initialize generators
        self.G_A2B = Generator().to(self.device)
        self.G_B2A = Generator().to(self.device)
        
        # Initialize discriminators
        self.D_A = Discriminator().to(self.device)
        self.D_B = Discriminator().to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(
            list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()),
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
        
    def train_step(self, real_A, real_B):
        """
        Perform one training step
        
        Args:
            real_A: Mel-spectrogram from domain A
            real_B: Mel-spectrogram from domain B
            
        Returns:
            dict: Dictionary containing loss values
        """
        # Generate fake samples
        fake_B = self.G_A2B(real_A)
        fake_A = self.G_B2A(real_B)
        
        # Reconstruct samples
        rec_A = self.G_B2A(fake_B)
        rec_B = self.G_A2B(fake_A)
        
        # Identity mapping
        idt_A = self.G_B2A(real_A)
        idt_B = self.G_A2B(real_B)
        
        # Train Generators
        self.g_optimizer.zero_grad()
        
        # GAN loss
        loss_GAN_A2B = self.criterion_GAN(self.D_B(fake_B), torch.ones_like(self.D_B(fake_B)))
        loss_GAN_B2A = self.criterion_GAN(self.D_A(fake_A), torch.ones_like(self.D_A(fake_A)))
        loss_GAN = loss_GAN_A2B + loss_GAN_B2A
        
        # Cycle loss
        loss_cycle_A = self.criterion_cycle(rec_A, real_A)
        loss_cycle_B = self.criterion_cycle(rec_B, real_B)
        loss_cycle = loss_cycle_A + loss_cycle_B
        
        # Identity loss
        loss_identity_A = self.criterion_identity(idt_A, real_A)
        loss_identity_B = self.criterion_identity(idt_B, real_B)
        loss_identity = loss_identity_A + loss_identity_B
        
        # Total generator loss
        loss_G = (loss_GAN + 
                 self.config['lambda_cycle'] * loss_cycle +
                 self.config['lambda_identity'] * loss_identity)
        
        loss_G.backward()
        self.g_optimizer.step()
        
        # Train Discriminators
        self.d_optimizer.zero_grad()
        
        # Real loss
        loss_real_A = self.criterion_GAN(self.D_A(real_A), torch.ones_like(self.D_A(real_A)))
        loss_real_B = self.criterion_GAN(self.D_B(real_B), torch.ones_like(self.D_B(real_B)))
        
        # Fake loss
        loss_fake_A = self.criterion_GAN(self.D_A(fake_A.detach()), torch.zeros_like(self.D_A(fake_A.detach())))
        loss_fake_B = self.criterion_GAN(self.D_B(fake_B.detach()), torch.zeros_like(self.D_B(fake_B.detach())))
        
        # Total discriminator loss
        loss_D_A = (loss_real_A + loss_fake_A) * 0.5
        loss_D_B = (loss_real_B + loss_fake_B) * 0.5
        loss_D = loss_D_A + loss_D_B
        
        loss_D.backward()
        self.d_optimizer.step()
        
        return {
            'G_loss': loss_G.item(),
            'D_loss': loss_D.item(),
            'G_adv': loss_GAN.item(),
            'cycle_loss': loss_cycle.item(),
            'identity_loss': loss_identity.item()
        }
    
    def convert_voice(self, mel_spectrogram, source_domain='A'):
        """
        Convert voice between domains
        
        Args:
            mel_spectrogram: Input mel-spectrogram
            source_domain: Source domain ('A' or 'B')
            
        Returns:
            Tensor: Converted mel-spectrogram
        """
        self.G_A2B.eval()
        self.G_B2A.eval()
        
        with torch.no_grad():
            if source_domain == 'A':
                return self.G_A2B(mel_spectrogram)
            else:
                return self.G_B2A(mel_spectrogram)
    
    def save_models(self, path):
        """Save model checkpoints"""
        torch.save({
            'G_A2B_state_dict': self.G_A2B.state_dict(),
            'G_B2A_state_dict': self.G_B2A.state_dict(),
            'D_A_state_dict': self.D_A.state_dict(),
            'D_B_state_dict': self.D_B.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
        }, path)
    
    def load_models(self, path):
        """Load model checkpoints"""
        checkpoint = torch.load(path)
        self.G_A2B.load_state_dict(checkpoint['G_A2B_state_dict'])
        self.G_B2A.load_state_dict(checkpoint['G_B2A_state_dict'])
        self.D_A.load_state_dict(checkpoint['D_A_state_dict'])
        self.D_B.load_state_dict(checkpoint['D_B_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict']) 