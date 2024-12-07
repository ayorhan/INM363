"""
FUNIT: Few-Shot Unsupervised Image-to-Image Translation

Source: https://arxiv.org/abs/1905.01723
Type: GAN (Generative Adversarial Network)

Architecture:
- Class-conditional generator
- Multi-task discriminator
- Few-shot adaptation
- Content encoder
- Style encoder

Pros:
- Few-shot learning capability
- Unsupervised training
- Flexible style adaptation
- Good generalization

Cons:
- Complex training setup
- Limited by few-shot samples
- Style consistency issues
- Resource intensive
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class ContentEncoder(nn.Module):
    """
    Content Encoder for FUNIT
    Extracts content features while maintaining spatial information
    """
    def __init__(self, input_dim=3, dim=64, n_downsample=2, n_res=4):
        super(ContentEncoder, self).__init__()
        
        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(input_dim, dim, 7)),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling blocks
        for _ in range(n_downsample):
            layers += [
                spectral_norm(nn.Conv2d(dim, dim*2, 4, stride=2, padding=1)),
                nn.InstanceNorm2d(dim*2),
                nn.ReLU(inplace=True)
            ]
            dim *= 2
            
        # Residual blocks
        for _ in range(n_res):
            layers += [ResidualBlock(dim)]
            
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class ClassEncoder(nn.Module):
    """
    Class Encoder for FUNIT
    Extracts class-specific style information
    """
    def __init__(self, input_dim=3, dim=64, n_downsample=4, n_class=10):
        super(ClassEncoder, self).__init__()
        
        layers = [
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(input_dim, dim, 7)),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling blocks
        for _ in range(n_downsample):
            layers += [
                spectral_norm(nn.Conv2d(dim, dim*2, 4, stride=2, padding=1)),
                nn.ReLU(inplace=True)
            ]
            dim *= 2
            
        # Global average pooling and class prediction
        self.model = nn.Sequential(*layers)
        self.class_pred = nn.Linear(dim, n_class)
        self.style = nn.Linear(dim, dim)
        
    def forward(self, x):
        features = self.model(x)
        pooled = F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
        class_pred = self.class_pred(pooled)
        style = self.style(pooled)
        return class_pred, style

class MLP(nn.Module):
    """
    MLP for style processing
    """
    def __init__(self, input_dim, output_dim, dim=256, n_layers=3):
        super(MLP, self).__init__()
        
        layers = [nn.Linear(input_dim, dim),
                 nn.ReLU(inplace=True)]
        
        for _ in range(n_layers - 2):
            layers += [nn.Linear(dim, dim),
                      nn.ReLU(inplace=True)]
            
        layers += [nn.Linear(dim, output_dim)]
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class AdaptiveInstanceNorm(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) layer
    """
    def __init__(self, style_dim, num_features):
        super(AdaptiveInstanceNorm, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.mlp = MLP(style_dim, num_features * 2)
        
    def forward(self, content, style):
        style = self.mlp(style)
        gamma, beta = style.chunk(2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = self.norm(content)
        return gamma * out + beta

class ResidualBlock(nn.Module):
    """
    Residual Block with AdaIN
    """
    def __init__(self, dim, style_dim=None):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = spectral_norm(nn.Conv2d(dim, dim, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(dim, dim, 3, padding=1))
        
        if style_dim is not None:
            self.norm1 = AdaptiveInstanceNorm(style_dim, dim)
            self.norm2 = AdaptiveInstanceNorm(style_dim, dim)
        else:
            self.norm1 = nn.InstanceNorm2d(dim)
            self.norm2 = nn.InstanceNorm2d(dim)
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, s=None):
        residual = x
        out = self.relu(self.norm1(self.conv1(x), s) if s is not None else self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out), s) if s is not None else self.norm2(self.conv2(out))
        return x + out

class Generator(nn.Module):
    """
    FUNIT Generator combining content and class features
    """
    def __init__(self, content_dim=256, style_dim=256, n_res=4):
        super(Generator, self).__init__()
        
        # Residual blocks with AdaIN
        layers = []
        for _ in range(n_res):
            layers += [ResidualBlock(content_dim, style_dim)]
            
        # Upsampling blocks
        for _ in range(2):
            layers += [
                nn.Upsample(scale_factor=2),
                spectral_norm(nn.Conv2d(content_dim, content_dim//2, 5, padding=2)),
                AdaptiveInstanceNorm(style_dim, content_dim//2),
                nn.ReLU(inplace=True)
            ]
            content_dim //= 2
            
        # Output layer
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(content_dim, 3, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, content, style):
        return self.model(content, style)

class Discriminator(nn.Module):
    """
    Multi-task Discriminator for FUNIT
    """
    def __init__(self, input_dim=3, dim=64, n_class=10):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1))]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(input_dim, dim, normalize=False),
            *discriminator_block(dim, dim*2),
            *discriminator_block(dim*2, dim*4),
            *discriminator_block(dim*4, dim*8),
        )
        
        # Output layers
        self.adv_layer = nn.Conv2d(dim*8, 1, 3, padding=1)  # Real/Fake output
        self.cls_layer = nn.Conv2d(dim*8, n_class, 3, padding=1)  # Class prediction
        
    def forward(self, x):
        features = self.model(x)
        validity = self.adv_layer(features)
        class_pred = self.cls_layer(features)
        return validity, class_pred

class FUNIT:
    """
    FUNIT wrapper class for training and inference
    """
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Initialize networks
        self.content_encoder = ContentEncoder(
            input_dim=config['input_dim'],
            dim=config['dim']
        ).to(self.device)
        
        self.class_encoder = ClassEncoder(
            input_dim=config['input_dim'],
            dim=config['dim'],
            n_class=config['n_class']
        ).to(self.device)
        
        self.generator = Generator(
            content_dim=config['content_dim'],
            style_dim=config['style_dim']
        ).to(self.device)
        
        self.discriminator = Discriminator(
            input_dim=config['input_dim'],
            dim=config['dim'],
            n_class=config['n_class']
        ).to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(
            list(self.content_encoder.parameters()) +
            list(self.class_encoder.parameters()) +
            list(self.generator.parameters()),
            lr=config['g_lr'],
            betas=(config['beta1'], config['beta2'])
        )
        
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config['d_lr'],
            betas=(config['beta1'], config['beta2'])
        )
        
    def train_step(self, content_images, class_images, class_labels):
        """Perform one training step"""
        # Extract content and class features
        content_features = self.content_encoder(content_images)
        class_pred, style = self.class_encoder(class_images)
        
        # Generate images
        generated_images = self.generator(content_features, style)
        
        # Discriminator forward pass
        real_validity, real_class = self.discriminator(class_images)
        fake_validity, fake_class = self.discriminator(generated_images.detach())
        
        # Calculate discriminator losses
        d_adv_loss = (F.mse_loss(real_validity, torch.ones_like(real_validity)) +
                     F.mse_loss(fake_validity, torch.zeros_like(fake_validity))) / 2
        d_cls_loss = F.cross_entropy(real_class, class_labels)
        
        d_loss = d_adv_loss + self.config['lambda_cls'] * d_cls_loss
        
        # Update discriminator
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()
        
        # Generator forward pass
        fake_validity, fake_class = self.discriminator(generated_images)
        
        # Calculate generator losses
        g_adv_loss = F.mse_loss(fake_validity, torch.ones_like(fake_validity))
        g_cls_loss = F.cross_entropy(fake_class, class_labels)
        g_rec_loss = F.l1_loss(generated_images, content_images)
        
        g_loss = (g_adv_loss + 
                 self.config['lambda_cls'] * g_cls_loss + 
                 self.config['lambda_rec'] * g_rec_loss)
        
        # Update generator
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'd_adv_loss': d_adv_loss.item(),
            'd_cls_loss': d_cls_loss.item(),
            'g_adv_loss': g_adv_loss.item(),
            'g_cls_loss': g_cls_loss.item(),
            'g_rec_loss': g_rec_loss.item()
        }
        
    def transfer_style(self, content_image, style_images, k_shot=1):
        """
        Perform few-shot style transfer
        
        Args:
            content_image: Source image to transfer style to
            style_images: K example images of target style
            k_shot: Number of style examples to use
        """
        self.eval()
        with torch.no_grad():
            # Extract content features
            content_features = self.content_encoder(content_image)
            
            # Extract and average style features from k examples
            style_features = []
            for style_image in style_images[:k_shot]:
                _, style = self.class_encoder(style_image)
                style_features.append(style)
            avg_style = torch.mean(torch.stack(style_features), dim=0)
            
            # Generate styled image
            return self.generator(content_features, avg_style) 