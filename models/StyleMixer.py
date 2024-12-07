"""
StyleMixer: Multi-Style Mixing Network

Source: Custom architecture for style mixing
Type: CNN (Convolutional Neural Network)

Architecture:
- Enhanced feature extractor
- Hierarchical style mixing
- Adaptive feature modulation
- Multi-scale processing
- Style attention mechanism

Pros:
- Handles multiple styles
- Smooth style interpolation
- Controllable mixing
- Good feature separation

Cons:
- Complex training process
- Memory intensive
- Requires style weight tuning
- Can produce artifacts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class FeatureExtractor(nn.Module):
    """
    Enhanced feature extractor with hierarchical processing
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        
        # Create hierarchical feature extraction blocks
        self.level1 = nn.Sequential(*list(vgg)[:4])   # relu1_1
        self.level2 = nn.Sequential(*list(vgg)[4:9])  # relu2_1
        self.level3 = nn.Sequential(*list(vgg)[9:18]) # relu3_1
        self.level4 = nn.Sequential(*list(vgg)[18:27])# relu4_1
        
        # Feature enhancement modules
        self.enhance1 = FeatureEnhancement(64)
        self.enhance2 = FeatureEnhancement(128)
        self.enhance3 = FeatureEnhancement(256)
        self.enhance4 = FeatureEnhancement(512)
        
        # Freeze VGG parameters
        for param in self.parameters():
            if not isinstance(param, nn.Parameter):
                param.requires_grad = False
                
    def forward(self, x):
        feat1 = self.enhance1(self.level1(x))
        feat2 = self.enhance2(self.level2(feat1))
        feat3 = self.enhance3(self.level3(feat2))
        feat4 = self.enhance4(self.level4(feat3))
        return [feat1, feat2, feat3, feat4]

class FeatureEnhancement(nn.Module):
    """
    Feature enhancement module with attention
    """
    def __init__(self, channels):
        super(FeatureEnhancement, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        att = self.attention(x)
        enhanced = x * att
        return self.conv(enhanced)

class StyleMixingModule(nn.Module):
    """
    Advanced style mixing module with attention and region awareness
    """
    def __init__(self, channels):
        super(StyleMixingModule, self).__init__()
        
        # Style attention layers
        self.style_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels//4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels//4, channels, 1),
                nn.Sigmoid()
            ) for _ in range(4)  # For different style aspects
        ])
        
        # Region awareness module
        self.region_awareness = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, padding=1),
            nn.InstanceNorm2d(channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, 4, 1),  # 4 regions
            nn.Softmax(dim=1)
        )
        
        # Style fusion module
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, content_feat, style_feats):
        batch_size = content_feat.size(0)
        
        # Apply style attention to each style
        attended_styles = []
        for style_feat in style_feats:
            style_aspects = []
            for attention in self.style_attention:
                att = attention(style_feat)
                style_aspects.append(style_feat * att)
            attended_styles.append(torch.stack(style_aspects))
        
        # Compute region awareness
        regions = self.region_awareness(content_feat)
        
        # Mix styles based on regions and attention
        mixed_features = []
        for i in range(4):  # For each region
            region_mask = regions[:, i:i+1]
            region_styles = []
            
            for attended_style in attended_styles:
                region_style = attended_style[i] * region_mask
                region_styles.append(region_style)
            
            mixed_features.append(sum(region_styles))
        
        # Fuse mixed features
        mixed = torch.cat(mixed_features, dim=1)
        return self.fusion(mixed)

class AdaptiveStyleTransfer(nn.Module):
    """
    Adaptive style transfer module with enhanced normalization
    """
    def __init__(self, channels, eps=1e-5):
        super(AdaptiveStyleTransfer, self).__init__()
        self.eps = eps
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Style modulation layers
        self.modulation = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, 1)
        )
        
    def forward(self, content_feat, style_feat):
        # Calculate statistics
        c_mean, c_std = self.calc_mean_std(content_feat)
        s_mean, s_std = self.calc_mean_std(style_feat)
        
        # Compute modulation parameters
        combined = torch.cat([content_feat, style_feat], dim=1)
        mod_params = self.modulation(combined)
        gamma, beta = mod_params.chunk(2, dim=1)
        
        # Apply adaptive style transfer
        normalized = (content_feat - c_mean) / c_std
        transformed = normalized * (gamma * s_std) + (beta * s_mean)
        
        return self.weight * transformed + self.bias
    
    def calc_mean_std(self, feat):
        mean = feat.mean((2, 3), keepdim=True)
        std = (feat.var((2, 3), keepdim=True) + self.eps).sqrt()
        return mean, std

class Decoder(nn.Module):
    """
    Decoder network for reconstructing images from features
    """
    def __init__(self):
        super(Decoder, self).__init__()
        # Upsampling blocks matching encoder levels
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, features):
        x = self.up4(features[3])
        x = F.interpolate(x, scale_factor=2)
        x = self.up3(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.up2(x)
        x = F.interpolate(x, scale_factor=2)
        return self.up1(x)

class StyleMixerModel:
    """
    Complete StyleMixer model for flexible style mixing
    """
    def __init__(self, config):
        self.device = config['device']
        self.config = config
        
        # Initialize networks
        self.encoder = FeatureExtractor().to(self.device)
        self.decoder = Decoder().to(self.device)
        self.style_mixer = nn.ModuleList([
            StyleMixingModule(64).to(self.device),
            StyleMixingModule(128).to(self.device),
            StyleMixingModule(256).to(self.device),
            StyleMixingModule(512).to(self.device)
        ])
        self.adaptive_transfer = nn.ModuleList([
            AdaptiveStyleTransfer(64).to(self.device),
            AdaptiveStyleTransfer(128).to(self.device),
            AdaptiveStyleTransfer(256).to(self.device),
            AdaptiveStyleTransfer(512).to(self.device)
        ])
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.decoder.parameters()) +
            list(self.style_mixer.parameters()) +
            list(self.adaptive_transfer.parameters()),
            lr=config['lr']
        )
        
    def mix_styles(self, content_image, style_images, style_weights=None):
        """
        Mix multiple styles with content
        
        Args:
            content_image: Content image tensor
            style_images: List of style image tensors
            style_weights: Optional weights for style mixing
            
        Returns:
            Tensor: Style-mixed image
        """
        # Extract features
        content_features = self.encoder(content_image)
        style_features = [self.encoder(style) for style in style_images]
        
        # Process features at each level
        mixed_features = []
        for level in range(len(content_features)):
            # Get features at current level
            c_feat = content_features[level]
            s_feats = [s[level] for s in style_features]
            
            # Mix styles
            mixed_style = self.style_mixer[level](c_feat, s_feats)
            
            # Apply adaptive transfer
            transferred = self.adaptive_transfer[level](c_feat, mixed_style)
            
            mixed_features.append(transferred)
        
        # Decode features
        return self.decoder(mixed_features)
    
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
        
        # Generate mixed-style images
        mixed = self.mix_styles(content_images, style_images)
        
        # Compute losses
        content_loss = self.compute_content_loss(
            mixed, content_images)
        style_loss = self.compute_style_loss(
            mixed, style_images)
        
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