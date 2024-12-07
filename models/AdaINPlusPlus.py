"""
AdaIN++: Enhanced Adaptive Instance Normalization for Style Transfer

Source: Builds on AdaIN (https://arxiv.org/abs/1703.06868)
Type: CNN (Convolutional Neural Network)

Architecture:
- VGG-based encoder with enhancement blocks
- Multi-level feature processing
- Self-attention modules for improved style transfer
- Enhanced decoder with skip connections
- Style mixing module for multiple style inputs

Pros:
- Improved style transfer quality over original AdaIN
- Better preservation of content structure
- Handles multiple style inputs
- Fast inference time

Cons:
- Higher memory requirements than basic AdaIN
- May struggle with extreme style changes
- Limited by VGG feature space
- Training requires significant data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class VGGEncoder(nn.Module):
    """
    Enhanced VGG encoder for AdaIN++
    Extracts multi-level features with additional processing
    """
    def __init__(self):
        super(VGGEncoder, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        
        # Create encoder blocks with enhanced feature extraction
        self.slice1 = nn.Sequential(*list(vgg)[:4])   # relu1_1
        self.slice2 = nn.Sequential(*list(vgg)[4:9])  # relu2_1
        self.slice3 = nn.Sequential(*list(vgg)[9:18]) # relu3_1
        self.slice4 = nn.Sequential(*list(vgg)[18:27])# relu4_1
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # Additional feature processing layers
        self.enhancement = nn.ModuleList([
            EnhancementBlock(64),   # For slice1
            EnhancementBlock(128),  # For slice2
            EnhancementBlock(256),  # For slice3
            EnhancementBlock(512)   # For slice4
        ])
        
    def forward(self, x):
        h1 = self.enhancement[0](self.slice1(x))
        h2 = self.enhancement[1](self.slice2(h1))
        h3 = self.enhancement[2](self.slice3(h2))
        h4 = self.enhancement[3](self.slice4(h3))
        return [h1, h2, h3, h4]

class EnhancementBlock(nn.Module):
    """
    Feature enhancement block for improved style transfer
    """
    def __init__(self, channels):
        super(EnhancementBlock, self).__init__()
        self.attention = SelfAttention(channels)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(self.attention(x))

class SelfAttention(nn.Module):
    """
    Self-attention module for feature enhancement
    """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
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

class AdaINPlusPlus(nn.Module):
    """
    Enhanced Adaptive Instance Normalization module
    """
    def __init__(self, epsilon=1e-5):
        super(AdaINPlusPlus, self).__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def calc_mean_std(self, feat, eps=1e-5):
        size = feat.size()
        mean = feat.view(*size[:2], -1).mean(dim=2).view(*size[:2], 1, 1)
        std = (feat.view(*size[:2], -1).var(dim=2) + eps).sqrt().view(*size[:2], 1, 1)
        return mean, std
    
    def forward(self, content_feat, style_feat):
        size = content_feat.size()
        
        # Calculate statistics
        c_mean, c_std = self.calc_mean_std(content_feat, self.epsilon)
        s_mean, s_std = self.calc_mean_std(style_feat, self.epsilon)
        
        # Normalize and adapt
        normalized_feat = (content_feat - c_mean) / c_std
        adapted_feat = normalized_feat * s_std + s_mean
        
        # Apply learnable parameters
        return self.weight * adapted_feat + self.bias

class StyleMixer(nn.Module):
    """
    Style mixing module for combining multiple styles
    """
    def __init__(self, channels):
        super(StyleMixer, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        weights = [self.attention(feat) for feat in features]
        weights = torch.stack(weights)
        weights = F.softmax(weights, dim=0)
        
        mixed = sum(w * f for w, f in zip(weights, features))
        return mixed

class Decoder(nn.Module):
    """
    Enhanced decoder for AdaIN++
    """
    def __init__(self):
        super(Decoder, self).__init__()
        
        # Decoder layers with residual connections
        self.ups = nn.ModuleList([
            UpBlock(512, 256),
            UpBlock(256, 128),
            UpBlock(128, 64),
            UpBlock(64, 3)
        ])
        
        # Final refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, features):
        x = features[-1]
        for i, up in enumerate(self.ups):
            if i < len(features) - 1:
                x = up(x, features[-(i+2)])
            else:
                x = up(x)
        return self.refinement(x)

class UpBlock(nn.Module):
    """
    Upsampling block with skip connections
    """
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip=None):
        x = self.conv(x)
        if skip is not None:
            x = x + skip
        return x

class AdaINPlusPlusModel:
    """
    Complete AdaIN++ model for style transfer
    """
    def __init__(self, config):
        self.device = config['device']
        self.config = config
        
        # Initialize networks
        self.encoder = VGGEncoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        self.adain = AdaINPlusPlus().to(self.device)
        self.style_mixer = StyleMixer(512).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.decoder.parameters()) + 
            list(self.adain.parameters()) + 
            list(self.style_mixer.parameters()),
            lr=config['lr']
        )
        
    def style_transfer(self, content_images, style_images, alpha=1.0):
        """
        Perform style transfer
        
        Args:
            content_images: Content image tensor
            style_images: List of style image tensors
            alpha: Style weight factor
            
        Returns:
            Tensor: Stylized image
        """
        # Extract features
        content_features = self.encoder(content_images)
        style_features_list = [self.encoder(style_img) for style_img in style_images]
        
        # Process features at each level
        adapted_features = []
        for level in range(len(content_features)):
            # Get features at current level
            c_feat = content_features[level]
            s_feats = [s_features[level] for s_features in style_features_list]
            
            # Mix style features
            mixed_style = self.style_mixer(s_feats)
            
            # Apply AdaIN++
            adapted = self.adain(c_feat, mixed_style)
            
            # Interpolate between content and adapted features
            adapted_features.append(
                (1 - alpha) * c_feat + alpha * adapted
            )
        
        # Decode features
        stylized = self.decoder(adapted_features)
        return stylized
    
    def compute_loss(self, stylized_images, content_images, style_images):
        """Compute content and style losses"""
        # Extract features
        stylized_features = self.encoder(stylized_images)
        content_features = self.encoder(content_images)
        style_features_list = [self.encoder(style_img) for style_img in style_images]
        
        # Content loss
        content_loss = F.mse_loss(stylized_features[-1], content_features[-1])
        
        # Style loss
        style_loss = 0
        for s_features in style_features_list:
            for sf, stf in zip(s_features, stylized_features):
                s_mean, s_std = self.adain.calc_mean_std(sf)
                st_mean, st_std = self.adain.calc_mean_std(stf)
                style_loss += F.mse_loss(st_mean, s_mean) + F.mse_loss(st_std, s_std)
        
        style_loss /= len(style_features_list)
        
        return content_loss, style_loss
    
    def train_step(self, content_images, style_images):
        """
        Perform one training step
        
        Args:
            content_images: Batch of content images
            style_images: List of batches of style images
            
        Returns:
            dict: Dictionary containing loss values
        """
        self.optimizer.zero_grad()
        
        # Generate stylized images
        stylized = self.style_transfer(content_images, style_images)
        
        # Compute losses
        content_loss, style_loss = self.compute_loss(
            stylized, content_images, style_images)
        
        # Total loss
        total_loss = (self.config['content_weight'] * content_loss +
                     self.config['style_weight'] * style_loss)
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'content_loss': content_loss.item(),
            'style_loss': style_loss.item(),
            'total_loss': total_loss.item()
        } 