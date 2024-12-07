"""
FastPhotoStyle: Photorealistic Style Transfer

Source: https://arxiv.org/abs/1802.06474
Type: CNN (Convolutional Neural Network)

Architecture:
- VGG-based encoder with edge preservation
- PhotoEncoder for structural feature extraction
- Edge-aware smoothing modules
- Photorealistic decoder with refinement
- Multi-scale feature processing

Pros:
- Produces photorealistic results
- Preserves structural integrity
- Fast inference compared to optimization-based methods
- Good edge preservation

Cons:
- May lose fine texture details
- Limited by photorealism constraints
- Requires careful parameter tuning
- Can produce artifacts in complex scenes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class PhotoEncoder(nn.Module):
    """
    Encoder optimized for photorealistic style transfer
    """
    def __init__(self):
        super(PhotoEncoder, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        
        # Create encoder blocks with skip connections
        self.block1 = nn.Sequential(*list(vgg)[:4])   # relu1_1
        self.block2 = nn.Sequential(*list(vgg)[4:9])  # relu2_1
        self.block3 = nn.Sequential(*list(vgg)[9:18]) # relu3_1
        self.block4 = nn.Sequential(*list(vgg)[18:27])# relu4_1
        
        # Edge preservation modules
        self.edge1 = EdgePreservationModule(64)
        self.edge2 = EdgePreservationModule(128)
        self.edge3 = EdgePreservationModule(256)
        self.edge4 = EdgePreservationModule(512)
        
        # Freeze VGG parameters
        for param in self.parameters():
            if not isinstance(param, nn.Parameter):
                param.requires_grad = False
                
    def forward(self, x):
        # Extract features with edge preservation
        feat1 = self.edge1(self.block1(x))
        feat2 = self.edge2(self.block2(feat1))
        feat3 = self.edge3(self.block3(feat2))
        feat4 = self.edge4(self.block4(feat3))
        
        return [feat1, feat2, feat3, feat4]

class EdgePreservationModule(nn.Module):
    """
    Module for preserving structural edges during style transfer
    """
    def __init__(self, channels):
        super(EdgePreservationModule, self).__init__()
        
        # Edge detection
        self.edge_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        sobel_kernel = torch.tensor([[-1, -2, -1],
                                   [ 0,  0,  0],
                                   [ 1,  2,  1]], dtype=torch.float32)
        sobel_kernel = sobel_kernel.view(1, 1, 3, 3)
        self.edge_conv.weight.data = sobel_kernel.repeat(channels, channels, 1, 1)
        self.edge_conv.weight.requires_grad = False
        
        # Edge enhancement
        self.enhance = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        edges = self.edge_conv(x)
        enhanced = self.enhance(torch.cat([x, edges], dim=1))
        return enhanced

class PhotoStyleTransfer(nn.Module):
    """
    Photorealistic style transfer module
    """
    def __init__(self, channels, eps=1e-5):
        super(PhotoStyleTransfer, self).__init__()
        self.eps = eps
        
        # Style transformation
        self.transform = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels*2, 1)
        )
        
        # Structure preservation
        self.structure = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, content_feat, style_feat):
        # Calculate statistics
        c_mean, c_std = self.calc_mean_std(content_feat)
        s_mean, s_std = self.calc_mean_std(style_feat)
        
        # Transform features while preserving structure
        combined = torch.cat([content_feat, style_feat], dim=1)
        params = self.transform(combined)
        gamma, beta = params.chunk(2, dim=1)
        
        # Apply transformation with structure preservation
        normalized = (content_feat - c_mean) / c_std
        styled = normalized * (gamma * s_std) + (beta * s_mean)
        structure = self.structure(content_feat)
        
        return styled + structure
    
    def calc_mean_std(self, feat):
        mean = feat.mean((2, 3), keepdim=True)
        std = (feat.var((2, 3), keepdim=True) + self.eps).sqrt()
        return mean, std

class PhotoDecoder(nn.Module):
    """
    Decoder optimized for photorealistic output
    """
    def __init__(self):
        super(PhotoDecoder, self).__init__()
        
        # Upsampling blocks with refinement
        self.up4 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up1 = UpBlock(64, 32)
        
        # Final refinement for photorealism
        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # Edge-aware smoothing
        self.smooth = EdgeAwareSmoothing()
        
    def forward(self, features):
        # Decode features
        x = self.up4(features[3])
        x = self.up3(x + features[2])
        x = self.up2(x + features[1])
        x = self.up1(x + features[0])
        
        # Final refinement with edge preservation
        x = self.refine(x)
        x = self.smooth(x)
        return x

class UpBlock(nn.Module):
    """
    Upsampling block with edge awareness
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
        
        self.edge_attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        feat = self.conv(x)
        attention = self.edge_attention(feat)
        return feat * attention

class EdgeAwareSmoothing(nn.Module):
    """
    Edge-aware smoothing module for maintaining photorealism
    """
    def __init__(self):
        super(EdgeAwareSmoothing, self).__init__()
        
        # Guided filter parameters
        self.r = 1  # Filter radius
        self.eps = 1e-4  # Regularization
        
        # Edge detection
        self.edge_detect = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        sobel_kernel = torch.tensor([[-1, -2, -1],
                                   [ 0,  0,  0],
                                   [ 1,  2,  1]], dtype=torch.float32)
        sobel_kernel = sobel_kernel.view(1, 1, 3, 3)
        self.edge_detect.weight.data = sobel_kernel.repeat(3, 3, 1, 1)
        self.edge_detect.weight.requires_grad = False
        
    def forward(self, x):
        # Detect edges
        edges = self.edge_detect(x)
        
        # Apply guided filtering
        return self.guided_filter(x, edges)
    
    def guided_filter(self, x, guidance):
        """Apply guided filter for edge-aware smoothing"""
        N = self.box_filter(torch.ones_like(x))
        
        mean_x = self.box_filter(x) / N
        mean_g = self.box_filter(guidance) / N
        cov_xg = self.box_filter(x * guidance) / N - mean_x * mean_g
        var_g = self.box_filter(guidance * guidance) / N - mean_g * mean_g
        
        A = cov_xg / (var_g + self.eps)
        b = mean_x - A * mean_g
        
        mean_A = self.box_filter(A) / N
        mean_b = self.box_filter(b) / N
        
        return mean_A * guidance + mean_b
    
    def box_filter(self, x):
        """Fast box filtering implementation"""
        kernel_size = 2 * self.r + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size).to(x.device) / (kernel_size ** 2)
        return F.conv2d(x, kernel.repeat(x.size(1), 1, 1, 1), padding=self.r, groups=x.size(1))

class FastPhotoStyle:
    """
    Complete FastPhotoStyle model
    """
    def __init__(self, config):
        self.device = config['device']
        self.config = config
        
        # Initialize networks
        self.encoder = PhotoEncoder().to(self.device)
        self.decoder = PhotoDecoder().to(self.device)
        self.transfer = nn.ModuleList([
            PhotoStyleTransfer(64).to(self.device),
            PhotoStyleTransfer(128).to(self.device),
            PhotoStyleTransfer(256).to(self.device),
            PhotoStyleTransfer(512).to(self.device)
        ])
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.decoder.parameters()) +
            list(self.transfer.parameters()),
            lr=config['lr']
        )
        
    def style_transfer(self, content_image, style_image):
        """
        Perform photorealistic style transfer
        
        Args:
            content_image: Content image tensor
            style_image: Style image tensor
            
        Returns:
            Tensor: Stylized photorealistic image
        """
        # Extract features
        content_features = self.encoder(content_image)
        style_features = self.encoder(style_image)
        
        # Apply style transfer at each level
        transferred_features = []
        for i in range(len(content_features)):
            transferred = self.transfer[i](
                content_features[i], style_features[i])
            transferred_features.append(transferred)
        
        # Decode features
        return self.decoder(transferred_features)
    
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
        
        # Compute losses
        content_loss = self.compute_content_loss(stylized, content_images)
        style_loss = self.compute_style_loss(stylized, style_images)
        photorealism_loss = self.compute_photorealism_loss(
            stylized, content_images)
        
        # Total loss
        total_loss = (self.config['content_weight'] * content_loss +
                     self.config['style_weight'] * style_loss +
                     self.config['photorealism_weight'] * photorealism_loss)
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'content_loss': content_loss.item(),
            'style_loss': style_loss.item(),
            'photorealism_loss': photorealism_loss.item(),
            'total_loss': total_loss.item()
        } 