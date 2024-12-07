"""
ReStyle: A Residual-Based StyleGAN Encoder

Source: https://arxiv.org/abs/2104.02699
Type: GAN (Generative Adversarial Network)

Architecture:
- Iterative refinement
- StyleGAN-based encoder
- Residual learning
- Progressive encoding
- Feature alignment

Pros:
- High-quality image inversion
- Stable encoding
- Good identity preservation
- Flexible manipulation

Cons:
- Iterative computation overhead
- Large model size
- Complex training process
- GPU memory intensive
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return x + residual

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.up(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, skip):
        combined = torch.cat([x, skip], dim=1)
        attention_weights = self.attention(combined)
        return x + skip * attention_weights

class StyleTransformBlock(nn.Module):
    def __init__(self, channels):
        super(StyleTransformBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class ReStyle(nn.Module):
    """
    ReStyle: Iterative refinement for style transfer
    """
    def __init__(self, config):
        super(ReStyle, self).__init__()
        self.device = config['device']
        self.num_iterations = config.get('num_iterations', 5)
        
        # Initialize networks
        self.encoder = RefinementEncoder().to(self.device)
        self.decoder = RefinementDecoder().to(self.device)
        self.style_net = StyleNetwork().to(self.device)
        
        # Refinement modules
        self.refinement = nn.ModuleList([
            RefinementBlock().to(self.device)
            for _ in range(self.num_iterations)
        ])
        
        # VGG for perceptual loss
        self.vgg = VGGPerceptual().to(self.device)
        
    def forward(self, content_image, style_image):
        """
        Forward pass with iterative refinement
        """
        batch_size = content_image.size(0)
        current_result = content_image
        
        # Initial style encoding
        style_code = self.style_net(style_image)
        
        # Store intermediate results
        intermediate_results = []
        
        # Iterative refinement
        for i in range(self.num_iterations):
            # Encode current result
            content_features = self.encoder(current_result)
            
            # Apply refinement
            refined_features = self.refinement[i](content_features, style_code)
            
            # Decode refined features
            current_result = self.decoder(refined_features)
            intermediate_results.append(current_result)
            
        return current_result, intermediate_results

class RefinementEncoder(nn.Module):
    """
    Encoder with progressive refinement capabilities
    """
    def __init__(self):
        super(RefinementEncoder, self).__init__()
        
        # Convolutional layers
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        
        # Instance normalization
        self.norm = nn.InstanceNorm2d(512)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(512)
            for _ in range(4)
        ])
        
    def forward(self, x):
        # Progressive feature extraction
        features = []
        
        x = self.conv1(x)
        features.append(x)
        
        x = self.conv2(x)
        features.append(x)
        
        x = self.conv3(x)
        features.append(x)
        
        x = self.conv4(x)
        features.append(x)
        
        x = self.norm(x)
        
        # Apply residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
            
        return {'final': x, 'intermediate': features}

class RefinementDecoder(nn.Module):
    """
    Decoder with progressive upsampling
    """
    def __init__(self):
        super(RefinementDecoder, self).__init__()
        
        # Upsampling blocks
        self.up1 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up4 = UpBlock(64, 32)
        
        # Final convolution
        self.final = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # Refinement attention
        self.attention = nn.ModuleList([
            AttentionBlock(ch)
            for ch in [256, 128, 64, 32]
        ])
        
    def forward(self, features):
        x = features['final']
        intermediates = features['intermediate']
        
        # Progressive upsampling with attention
        x = self.up1(x)
        x = self.attention[0](x, intermediates[-1])
        
        x = self.up2(x)
        x = self.attention[1](x, intermediates[-2])
        
        x = self.up3(x)
        x = self.attention[2](x, intermediates[-3])
        
        x = self.up4(x)
        x = self.attention[3](x, intermediates[-4])
        
        return self.final(x)

class StyleNetwork(nn.Module):
    """
    Style encoding network
    """
    def __init__(self):
        super(StyleNetwork, self).__init__()
        
        # VGG-based feature extraction
        vgg = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg)[:4])
        self.slice2 = nn.Sequential(*list(vgg)[4:9])
        self.slice3 = nn.Sequential(*list(vgg)[9:18])
        self.slice4 = nn.Sequential(*list(vgg)[18:27])
        
        # Style transformation
        self.transform = nn.ModuleList([
            StyleTransformBlock(ch)
            for ch in [64, 128, 256, 512]
        ])
        
        # Freeze VGG parameters
        for param in self.parameters():
            if not isinstance(param, nn.Parameter):
                param.requires_grad = False
                
    def forward(self, x):
        style_features = []
        
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        
        # Transform style features
        style_features.append(self.transform[0](h1))
        style_features.append(self.transform[1](h2))
        style_features.append(self.transform[2](h3))
        style_features.append(self.transform[3](h4))
        
        return style_features

class RefinementBlock(nn.Module):
    """
    Feature refinement block
    """
    def __init__(self):
        super(RefinementBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(512, 512, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(512)
        
        self.conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(512)
        
        # Style modulation
        self.style_mod = AdaptiveStyleMod(512)
        
    def forward(self, content_features, style_code):
        x = content_features['final']
        
        # First refinement
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        
        # Style modulation
        x = self.style_mod(x, style_code[-1])
        
        # Second refinement
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        
        return {'final': x + residual, 'intermediate': content_features['intermediate']}

class AdaptiveStyleMod(nn.Module):
    """
    Adaptive style modulation
    """
    def __init__(self, channels):
        super(AdaptiveStyleMod, self).__init__()
        
        self.conv = nn.Conv2d(channels*2, channels, 1)
        self.norm = nn.InstanceNorm2d(channels)
        
    def forward(self, content, style):
        # Compute statistics
        mean = style.mean([2, 3], keepdim=True)
        std = style.std([2, 3], keepdim=True) + 1e-6
        
        # Normalize content
        normalized = (content - content.mean([2, 3], keepdim=True)) / (content.std([2, 3], keepdim=True) + 1e-6)
        
        # Apply style
        styled = normalized * std + mean
        
        # Combine features
        combined = torch.cat([content, styled], dim=1)
        return self.norm(self.conv(combined))

class VGGPerceptual(nn.Module):
    """VGG-based perceptual loss network"""
    def __init__(self):
        super(VGGPerceptual, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        
        self.content_layer = nn.Sequential(*list(vgg)[:36])  # conv5_4
        self.style_layers = nn.ModuleList([
            nn.Sequential(*list(vgg)[:4]),   # conv1_2
            nn.Sequential(*list(vgg)[:9]),   # conv2_2
            nn.Sequential(*list(vgg)[:18]),  # conv3_4
            nn.Sequential(*list(vgg)[:27]),  # conv4_4
        ])
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        content = self.content_layer(x)
        style = [layer(x) for layer in self.style_layers]
        return {'content': content, 'style': style}

class ReStyleModel:
    """
    Complete ReStyle model with training and inference
    """
    def __init__(self, config):
        self.device = config['device']
        self.config = config
        
        # Initialize ReStyle
        self.model = ReStyle(config).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['lr']
        )
        
    def train_step(self, content_images, style_images):
        """
        Perform one training step
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        output, intermediates = self.model(content_images, style_images)
        
        # Compute losses
        content_loss = self.compute_content_loss(
            output, content_images)
        style_loss = self.compute_style_loss(
            output, style_images)
        refinement_loss = self.compute_refinement_loss(
            intermediates)
        
        # Total loss
        total_loss = (self.config['content_weight'] * content_loss +
                     self.config['style_weight'] * style_loss +
                     self.config['refinement_weight'] * refinement_loss)
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'content_loss': content_loss.item(),
            'style_loss': style_loss.item(),
            'refinement_loss': refinement_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def compute_content_loss(self, output, content_images):
        """Compute content loss"""
        content_features = self.model.vgg(content_images)
        output_features = self.model.vgg(output)
        
        return F.mse_loss(
            output_features['content'],
            content_features['content']
        )
    
    def compute_style_loss(self, output, style_images):
        """Compute style loss"""
        style_features = self.model.vgg(style_images)
        output_features = self.model.vgg(output)
        
        style_loss = 0
        for sf, of in zip(style_features['style'], output_features['style']):
            style_loss += F.mse_loss(
                self.gram_matrix(of),
                self.gram_matrix(sf)
            )
            
        return style_loss
    
    def compute_refinement_loss(self, intermediates):
        """Compute refinement consistency loss"""
        refinement_loss = 0
        for i in range(len(intermediates)-1):
            refinement_loss += F.l1_loss(
                intermediates[i],
                intermediates[i+1]
            )
        return refinement_loss
    
    def gram_matrix(self, x):
        """Compute gram matrix"""
        b, c, h, w = x.size()
        features = x.view(b, c, -1)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def style_transfer(self, content_image, style_image):
        """
        Perform style transfer
        """
        self.model.eval()
        with torch.no_grad():
            output, _ = self.model(content_image, style_image)
        return output 