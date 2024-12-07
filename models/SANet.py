"""
SANet: Arbitrary Style Transfer with Style-Attentional Networks

Source: https://arxiv.org/abs/1812.02342
Type: CNN (Convolutional Neural Network)

Architecture:
- Style-attentional networks
- Feature transformation
- Multi-head attention
- Style-based modulation
- Feature alignment

Pros:
- Arbitrary style transfer
- Better style-content fusion
- Attention-guided transfer
- Detail preservation

Cons:
- Computational overhead
- Memory intensive
- Complex attention mechanism
- Training stability issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class SAEncoder(nn.Module):
    """
    Style-Attentional Encoder with VGG backbone
    """
    def __init__(self):
        super(SAEncoder, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        
        # Create encoder blocks
        self.block1 = nn.Sequential(*list(vgg)[:4])   # relu1_1
        self.block2 = nn.Sequential(*list(vgg)[4:9])  # relu2_1
        self.block3 = nn.Sequential(*list(vgg)[9:18]) # relu3_1
        self.block4 = nn.Sequential(*list(vgg)[18:27])# relu4_1
        
        # Feature refinement modules
        self.refine1 = FeatureRefinement(64)
        self.refine2 = FeatureRefinement(128)
        self.refine3 = FeatureRefinement(256)
        self.refine4 = FeatureRefinement(512)
        
        # Freeze VGG parameters
        for param in self.parameters():
            if not isinstance(param, nn.Parameter):
                param.requires_grad = False
                
    def forward(self, x):
        feat1 = self.refine1(self.block1(x))
        feat2 = self.refine2(self.block2(feat1))
        feat3 = self.refine3(self.block3(feat2))
        feat4 = self.refine4(self.block4(feat3))
        return [feat1, feat2, feat3, feat4]

class FeatureRefinement(nn.Module):
    """
    Feature refinement module with channel attention
    """
    def __init__(self, channels):
        super(FeatureRefinement, self).__init__()
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        ca = self.channel_attention(x)
        sa = self.spatial_attention(x)
        refined = x * ca * sa
        return refined

class StyleAttentionModule(nn.Module):
    """
    Style attention module for feature transformation
    """
    def __init__(self, in_channels):
        super(StyleAttentionModule, self).__init__()
        self.in_channels = in_channels  # Store in_channels as instance variable
        
        self.query_conv = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Multi-head attention
        self.num_heads = 4
        self.head_dim = in_channels // 8 // self.num_heads
        
    def forward(self, content_feat, style_feat):
        batch_size = content_feat.size(0)
        
        # Multi-head attention
        attention_heads = []
        for _ in range(self.num_heads):
            # Query from content, Key/Value from style
            query = self.query_conv(content_feat).view(batch_size, -1, self.head_dim)
            key = self.key_conv(style_feat).view(batch_size, -1, self.head_dim)
            value = self.value_conv(style_feat).view(batch_size, -1, self.in_channels)
            
            # Compute attention scores
            attention = torch.bmm(query, key.transpose(1, 2))
            attention = F.softmax(attention / (self.head_dim ** 0.5), dim=-1)
            
            # Apply attention to values
            head_output = torch.bmm(attention, value)
            attention_heads.append(head_output)
        
        # Combine attention heads
        multi_head = torch.cat(attention_heads, dim=-1)
        multi_head = multi_head.view_as(content_feat)
        
        return self.gamma * multi_head + content_feat

class SADecoder(nn.Module):
    """
    Style-Attentional Decoder
    """
    def __init__(self):
        super(SADecoder, self).__init__()
        
        # Upsampling blocks
        self.up4 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up1 = UpBlock(64, 32)
        
        # Final convolution
        self.final = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # Style attention modules
        self.sa4 = StyleAttentionModule(512)
        self.sa3 = StyleAttentionModule(256)
        self.sa2 = StyleAttentionModule(128)
        self.sa1 = StyleAttentionModule(64)
        
    def forward(self, content_features, style_features):
        # Apply style attention at each level
        f4 = self.sa4(content_features[3], style_features[3])
        f3 = self.sa3(content_features[2], style_features[2])
        f2 = self.sa2(content_features[1], style_features[1])
        f1 = self.sa1(content_features[0], style_features[0])
        
        # Decode with skip connections
        x = self.up4(f4)
        x = self.up3(x + f3)
        x = self.up2(x + f2)
        x = self.up1(x + f1)
        
        return self.final(x)

class UpBlock(nn.Module):
    """
    Upsampling block with residual connection
    """
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.residual = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x + self.residual(x)

class SANet:
    """
    Complete Style-Attentional Network for style transfer
    """
    def __init__(self, config):
        self.device = config['device']
        self.config = config
        
        # Initialize networks
        self.encoder = SAEncoder().to(self.device)
        self.decoder = SADecoder().to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.decoder.parameters()),
            lr=config['lr']
        )
        
        # Loss functions
        self.content_criterion = nn.MSELoss()
        self.style_criterion = nn.MSELoss()
        
    def style_transfer(self, content_image, style_image):
        """
        Perform style transfer with attention
        
        Args:
            content_image: Content image tensor
            style_image: Style image tensor
            
        Returns:
            Tensor: Stylized image
        """
        # Extract features
        content_features = self.encoder(content_image)
        style_features = self.encoder(style_image)
        
        # Generate stylized image
        stylized = self.decoder(content_features, style_features)
        return stylized
    
    def compute_content_loss(self, stylized_features, content_features):
        """Compute content loss"""
        return sum(self.content_criterion(sf, cf) 
                  for sf, cf in zip(stylized_features, content_features))
    
    def compute_style_loss(self, stylized_features, style_features):
        """Compute style loss using attention-based gram matrices"""
        style_loss = 0
        for sf, stf in zip(style_features, stylized_features):
            # Compute attention-weighted gram matrices
            sf_gram = self.attention_gram_matrix(sf)
            stf_gram = self.attention_gram_matrix(stf)
            style_loss += self.style_criterion(stf_gram, sf_gram)
        return style_loss
    
    def attention_gram_matrix(self, x):
        """Compute attention-weighted gram matrix"""
        b, c, h, w = x.size()
        features = x.view(b, c, -1)
        attention = torch.bmm(features, features.transpose(1, 2))
        attention = F.softmax(attention, dim=-1)
        gram = torch.bmm(features, attention)
        return gram / (c * h * w)
    
    def train_step(self, content_images, style_images):
        """
        Perform one training step
        
        Args:
            content_images: Batch of content images
            style_images: Batch of style images
            
        Returns:
            dict: Dictionary containing loss values
        """
        self.optimizer.zero_grad()
        
        # Generate stylized images
        stylized = self.style_transfer(content_images, style_images)
        
        # Extract features for loss computation
        content_features = self.encoder(content_images)
        style_features = self.encoder(style_images)
        stylized_features = self.encoder(stylized)
        
        # Compute losses
        content_loss = self.compute_content_loss(
            stylized_features, content_features)
        style_loss = self.compute_style_loss(
            stylized_features, style_features)
        
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