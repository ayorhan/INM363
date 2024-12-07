"""
LST: Linear Style Transfer Network

Source: Custom architecture for efficient style transfer
Type: CNN (Convolutional Neural Network)

Architecture:
- Linear feature transformation
- Multi-layer style transfer
- Efficient encoder-decoder
- Content-style balancing
- Feature normalization

Pros:
- Fast inference
- Memory efficient
- Stable training
- Good style transfer quality

Cons:
- Limited by linear operations
- Less expressive than non-linear methods
- May miss complex patterns
- Style transfer limitations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class LinearEncoder(nn.Module):
    """
    Linear encoder for efficient style transfer
    """
    def __init__(self):
        super(LinearEncoder, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        
        # Create encoder blocks
        self.block1 = nn.Sequential(*list(vgg)[:4])   # relu1_1
        self.block2 = nn.Sequential(*list(vgg)[4:9])  # relu2_1
        self.block3 = nn.Sequential(*list(vgg)[9:18]) # relu3_1
        self.block4 = nn.Sequential(*list(vgg)[18:27])# relu4_1
        
        # Linear transformation modules
        self.linear1 = LinearTransform(64)
        self.linear2 = LinearTransform(128)
        self.linear3 = LinearTransform(256)
        self.linear4 = LinearTransform(512)
        
        # Freeze VGG parameters
        for param in self.parameters():
            if not isinstance(param, nn.Parameter):
                param.requires_grad = False
                
    def forward(self, x):
        feat1 = self.linear1(self.block1(x))
        feat2 = self.linear2(self.block2(feat1))
        feat3 = self.linear3(self.block3(feat2))
        feat4 = self.linear4(self.block4(feat3))
        return [feat1, feat2, feat3, feat4]

class LinearTransform(nn.Module):
    """
    Linear transformation module for feature processing
    """
    def __init__(self, channels):
        super(LinearTransform, self).__init__()
        
        self.linear = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
    def forward(self, x):
        transformed = self.linear(x)
        return transformed * self.scale + self.bias

class LinearStyleTransfer(nn.Module):
    """
    Linear style transfer module
    """
    def __init__(self, channels):
        super(LinearStyleTransfer, self).__init__()
        
        # Linear transformation matrices
        self.W = nn.Parameter(torch.eye(channels).unsqueeze(0))
        self.b = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
        # Feature alignment
        self.align = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )
        
    def forward(self, content_feat, style_feat):
        b, c, h, w = content_feat.size()
        
        # Reshape features
        content_flat = content_feat.view(b, c, -1)
        style_flat = style_feat.view(b, c, -1)
        
        # Linear transformation
        transformed = torch.bmm(self.W.expand(b, -1, -1), content_flat)
        transformed = transformed.view(b, c, h, w) + self.b
        
        # Feature alignment
        aligned = self.align(torch.cat([transformed, style_feat], dim=1))
        return aligned

class LinearDecoder(nn.Module):
    """
    Linear decoder for efficient reconstruction
    """
    def __init__(self):
        super(LinearDecoder, self).__init__()
        
        # Upsampling blocks
        self.up4 = LinearUpBlock(512, 256)
        self.up3 = LinearUpBlock(256, 128)
        self.up2 = LinearUpBlock(128, 64)
        self.up1 = LinearUpBlock(64, 32)
        
        # Final convolution
        self.final = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, features):
        x = self.up4(features[3])
        x = self.up3(x + features[2])
        x = self.up2(x + features[1])
        x = self.up1(x + features[0])
        return self.final(x)

class LinearUpBlock(nn.Module):
    """
    Linear upsampling block
    """
    def __init__(self, in_channels, out_channels):
        super(LinearUpBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.linear = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv(x)
        return self.linear(x)

class LST:
    """
    Complete Linear Style Transfer model
    """
    def __init__(self, config):
        self.device = config['device']
        self.config = config
        
        # Initialize networks
        self.encoder = LinearEncoder().to(self.device)
        self.decoder = LinearDecoder().to(self.device)
        self.transform = nn.ModuleList([
            LinearStyleTransfer(64).to(self.device),
            LinearStyleTransfer(128).to(self.device),
            LinearStyleTransfer(256).to(self.device),
            LinearStyleTransfer(512).to(self.device)
        ])
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.decoder.parameters()) +
            list(self.transform.parameters()),
            lr=config['lr']
        )
        
    def style_transfer(self, content_image, style_image):
        """
        Perform linear style transfer
        
        Args:
            content_image: Content image tensor
            style_image: Style image tensor
            
        Returns:
            Tensor: Stylized image
        """
        # Extract features
        content_features = self.encoder(content_image)
        style_features = self.encoder(style_image)
        
        # Apply linear transformations
        transformed_features = []
        for i in range(len(content_features)):
            transformed = self.transform[i](
                content_features[i], style_features[i])
            transformed_features.append(transformed)
        
        # Decode features
        return self.decoder(transformed_features)
    
    def compute_content_loss(self, stylized_features, content_features):
        """Compute content loss with linear weights"""
        content_loss = 0
        for i, (sf, cf) in enumerate(zip(stylized_features, content_features)):
            weight = self.config['content_weights'][i]
            content_loss += weight * F.mse_loss(sf, cf)
        return content_loss
    
    def compute_style_loss(self, stylized_features, style_features):
        """Compute style loss with linear gram matrices"""
        style_loss = 0
        for i, (sf, stf) in enumerate(zip(style_features, stylized_features)):
            weight = self.config['style_weights'][i]
            sf_gram = self.gram_matrix(sf)
            stf_gram = self.gram_matrix(stf)
            style_loss += weight * F.mse_loss(stf_gram, sf_gram)
        return style_loss
    
    def gram_matrix(self, x):
        """Compute gram matrix efficiently"""
        b, c, h, w = x.size()
        features = x.view(b, c, -1)
        gram = torch.bmm(features, features.transpose(1, 2))
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