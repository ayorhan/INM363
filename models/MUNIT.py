"""
MUNIT: Multimodal UNsupervised Image-to-image Translation

Source: https://arxiv.org/abs/1804.04732
Type: GAN (Generative Adversarial Network)

Architecture:
- Content-style disentanglement
- Multi-scale discriminators
- Adaptive instance normalization
- Style encoder network
- Content encoder network

Pros:
- Multiple style outputs
- Disentangled representations
- Good style diversity
- Flexible control

Cons:
- Complex training process
- High memory usage
- Sensitive to hyperparameters
- Potential style leakage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentEncoder(nn.Module):
    """Content encoder for domain-invariant content features"""
    def __init__(self, input_dim=3, dim=64, n_downsample=2, n_res=4):
        super(ContentEncoder, self).__init__()
        
        # Initial convolution
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_dim, dim, 7),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True)
        ]
        
        # Downsampling
        for i in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim*2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim*2),
                nn.ReLU(True)
            ]
            dim *= 2
            
        # Residual blocks
        for _ in range(n_res):
            layers += [ResidualBlock(dim)]
            
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class StyleEncoder(nn.Module):
    """Style encoder for domain-specific style features"""
    def __init__(self, input_dim=3, dim=64, style_dim=8, n_downsample=4):
        super(StyleEncoder, self).__init__()
        
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_dim, dim, 7),
            nn.ReLU(True)
        ]
        
        # Downsampling
        for i in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim*2, 4, stride=2, padding=1),
                nn.ReLU(True)
            ]
            dim *= 2
            
        # Global average pooling and FC
        layers += [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, style_dim, 1, 1, 0)
        ]
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    """Decoder with AdaIN for style-guided image generation"""
    def __init__(self, n_upsample=2, n_res=4, dim=256, output_dim=3, style_dim=8):
        super(Decoder, self).__init__()
        
        layers = []
        
        # Residual blocks with AdaIN
        for _ in range(n_res):
            layers += [AdaINResBlock(dim, style_dim)]
            
        # Upsampling
        for _ in range(n_upsample):
            layers += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(dim, dim//2, 5, stride=1, padding=2),
                AdaIN(dim//2, style_dim),
                nn.ReLU(True)
            ]
            dim //= 2
            
        # Output layer
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(dim, output_dim, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, content, style):
        return self.model((content, style))

class MUNIT(nn.Module):
    """Complete MUNIT model"""
    def __init__(self, config):
        super(MUNIT, self).__init__()
        self.config = config
        
        # Initialize networks
        self.content_encoder = ContentEncoder()
        self.style_encoder = StyleEncoder()
        self.decoder = Decoder()
        self.discriminator = MultiScaleDiscriminator()
        
        # Initialize optimizers
        self.gen_opt = torch.optim.Adam(
            list(self.content_encoder.parameters()) +
            list(self.style_encoder.parameters()) +
            list(self.decoder.parameters()),
            lr=config['lr'],
            betas=(0.5, 0.999)
        )
        self.dis_opt = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config['lr'],
            betas=(0.5, 0.999)
        )
        
    def forward(self, x_a, x_b):
        # Extract content and style
        c_a = self.content_encoder(x_a)
        s_a = self.style_encoder(x_a)
        c_b = self.content_encoder(x_b)
        s_b = self.style_encoder(x_b)
        
        # Decode cross-domain
        x_ba = self.decoder(c_b, s_a)
        x_ab = self.decoder(c_a, s_b)
        
        # Reconstruct
        c_b_recon = self.content_encoder(x_ba)
        s_a_recon = self.style_encoder(x_ba)
        c_a_recon = self.content_encoder(x_ab)
        s_b_recon = self.style_encoder(x_ab)
        
        x_a_recon = self.decoder(c_a, s_a)
        x_b_recon = self.decoder(c_b, s_b)
        
        return {
            'x_ab': x_ab, 'x_ba': x_ba,
            'x_a_recon': x_a_recon, 'x_b_recon': x_b_recon,
            'c_a': c_a, 'c_b': c_b,
            's_a': s_a, 's_b': s_b
        }
        
    def train_step(self, x_a, x_b):
        self.gen_opt.zero_grad()
        self.dis_opt.zero_grad()
        
        # Forward pass
        outputs = self.forward(x_a, x_b)
        
        # Compute losses
        recon_loss = self.compute_recon_loss(outputs)
        gen_loss = self.compute_gen_loss(outputs)
        dis_loss = self.compute_dis_loss(outputs)
        
        # Update generators
        g_loss = recon_loss + gen_loss
        g_loss.backward()
        self.gen_opt.step()
        
        # Update discriminator
        dis_loss.backward()
        self.dis_opt.step()
        
        return {
            'recon_loss': recon_loss.item(),
            'gen_loss': gen_loss.item(),
            'dis_loss': dis_loss.item()
        }

class AdaINResBlock(nn.Module):
    """Residual block with AdaIN"""
    def __init__(self, dim, style_dim):
        super(AdaINResBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.adain1 = AdaIN(dim, style_dim)
        self.adain2 = AdaIN(dim, style_dim)
        
    def forward(self, x):
        content, style = x
        out = self.conv1(content)
        out = self.adain1(out, style)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.adain2(out, style)
        return (content + out, style)

class AdaIN(nn.Module):
    """Adaptive Instance Normalization"""
    def __init__(self, dim, style_dim):
        super(AdaIN, self).__init__()
        self.fc = nn.Linear(style_dim, dim*2)
        
    def forward(self, x, style):
        style = self.fc(style)
        gamma, beta = style.chunk(2, 1)
        
        out = F.instance_norm(x)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        
        return gamma * out + beta

class Discriminator(nn.Module):
    """Single-scale discriminator"""
    def __init__(self, input_dim=3, dim=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, dim*2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(dim*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim*2, dim*4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(dim*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim*4, dim*8, 4, padding=1),
            nn.InstanceNorm2d(dim*8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim*8, 1, 4, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator"""
    def __init__(self, input_dim=3, dim=64, n_scales=3):
        super(MultiScaleDiscriminator, self).__init__()
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)
        self.discriminators = nn.ModuleList([
            Discriminator(input_dim, dim) for _ in range(n_scales)
        ])
        
    def forward(self, x):
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(x))
            x = self.downsample(x)
        return outputs 

class ResidualBlock(nn.Module):
    """Basic residual block"""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)