"""
PhotoWCT: Photorealistic Style Transfer via Wavelet Transforms

Source: https://arxiv.org/abs/1903.09760
Type: CNN (Convolutional Neural Network)

Architecture:
- Wavelet transform preprocessing
- VGG-based feature extraction
- Whitening and coloring transforms
- Multi-level stylization
- Photorealistic refinement

Pros:
- Photorealistic results
- Preserves structure
- Edge-aware transfer
- Robust to different styles

Cons:
- Slower than basic style transfer
- High memory usage
- Complex wavelet processing
- Parameter sensitivity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class VGGEncoder(nn.Module):
    """
    VGG-19 encoder modified for PhotoWCT
    Extracts multi-level features with skip connections
    """
    def __init__(self):
        super(VGGEncoder, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        
        # Create encoder blocks
        self.block1 = nn.Sequential(*list(vgg)[:4])   # relu1_1
        self.block2 = nn.Sequential(*list(vgg)[4:9])  # relu2_1
        self.block3 = nn.Sequential(*list(vgg)[9:18]) # relu3_1
        self.block4 = nn.Sequential(*list(vgg)[18:27])# relu4_1
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        feat1 = self.block1(x)
        feat2 = self.block2(feat1)
        feat3 = self.block3(feat2)
        feat4 = self.block4(feat3)
        return [feat1, feat2, feat3, feat4]

class VGGDecoder(nn.Module):
    """
    Symmetric decoder for VGG features
    Reconstructs images with unpooling layers
    """
    def __init__(self):
        super(VGGDecoder, self).__init__()
        
        # Decoder blocks with unpooling
        self.block4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features):
        feat4, feat3, feat2, feat1 = features
        x = self.block4(feat4)
        x = x + feat3
        x = self.block3(x)
        x = x + feat2
        x = self.block2(x)
        x = x + feat1
        x = self.block1(x)
        return x

class WCT(nn.Module):
    """
    Whitening and Coloring Transform
    Core component for style transfer
    """
    def __init__(self, eps=1e-5):
        super(WCT, self).__init__()
        self.eps = eps
        
    def whitening(self, content_feat):
        """Apply whitening transform to content features"""
        size = content_feat.size()
        content_flat = content_feat.view(size[0], size[1], -1)
        content_flat_mean = torch.mean(content_flat, dim=2, keepdim=True)
        content_flat_centered = content_flat - content_flat_mean
        
        # Compute covariance matrix
        content_cov = torch.bmm(content_flat_centered, 
                               content_flat_centered.transpose(1, 2))
        content_cov = content_cov / (size[2] * size[3] - 1) + torch.eye(
            size[1], device=content_feat.device).unsqueeze(0) * self.eps
        
        # SVD
        u, s, v = torch.svd(content_cov)
        
        # Whitening transform
        d = s.pow(-0.5)
        whitened = torch.bmm(torch.bmm(u, torch.diag_embed(d)), 
                            torch.bmm(v.transpose(1, 2), content_flat_centered))
        
        return whitened, u, s, v
    
    def coloring(self, whitened, style_feat, u, s, v):
        """Apply coloring transform using style statistics"""
        size = style_feat.size()
        style_flat = style_feat.view(size[0], size[1], -1)
        style_flat_mean = torch.mean(style_flat, dim=2, keepdim=True)
        style_flat_centered = style_flat - style_flat_mean
        
        # Compute style covariance
        style_cov = torch.bmm(style_flat_centered, 
                             style_flat_centered.transpose(1, 2))
        style_cov = style_cov / (size[2] * size[3] - 1) + torch.eye(
            size[1], device=style_feat.device).unsqueeze(0) * self.eps
        
        # SVD
        su, ss, sv = torch.svd(style_cov)
        
        # Coloring transform
        d = ss.pow(0.5)
        colored = torch.bmm(torch.bmm(su, torch.diag_embed(d)), 
                          torch.bmm(sv.transpose(1, 2), whitened))
        
        # Add style mean
        colored = colored + style_flat_mean
        colored = colored.view_as(style_feat)
        
        return colored

    def forward(self, content_feat, style_feat):
        """Apply WCT transform"""
        whitened, u, s, v = self.whitening(content_feat)
        colored = self.coloring(whitened, style_feat, u, s, v)
        return colored

class PhotoWCT(nn.Module):
    """
    PhotoWCT model for photorealistic style transfer
    Combines WCT with smoothing and preservation mechanisms
    """
    def __init__(self):
        super(PhotoWCT, self).__init__()
        self.encoder = VGGEncoder()
        self.decoder = VGGDecoder()
        self.wct = WCT()
        
        # Initialize smoothing filter
        kernel_size = 15
        sigma = 3.0
        kernel = self.create_gaussian_kernel(kernel_size, sigma)
        self.register_buffer('smooth_kernel', kernel)
        
    def create_gaussian_kernel(self, kernel_size, sigma):
        """Create 2D Gaussian kernel for smoothing"""
        x = torch.arange(kernel_size).float() - kernel_size // 2
        x = x.repeat(kernel_size).view(kernel_size, kernel_size)
        y = x.t()
        
        kernel = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        
        return kernel.view(1, 1, kernel_size, kernel_size)
    
    def smooth_filter(self, x):
        """Apply smoothing filter to maintain photorealism"""
        pad_size = self.smooth_kernel.size(-1) // 2
        x_pad = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        
        # Apply smoothing per channel
        channels = []
        for channel in range(x.size(1)):
            channel_data = x_pad[:, channel:channel+1]
            smoothed = F.conv2d(channel_data, self.smooth_kernel)
            channels.append(smoothed)
            
        return torch.cat(channels, dim=1)
    
    def forward(self, content, style, alpha=1.0):
        """
        Forward pass for photorealistic style transfer
        
        Args:
            content: Content image tensor
            style: Style image tensor
            alpha: Style weight factor
            
        Returns:
            Tensor: Stylized image
        """
        # Extract features
        content_features = self.encoder(content)
        style_features = self.encoder(style)
        
        # Apply WCT at multiple levels
        stylized_features = []
        for cf, sf in zip(content_features, style_features):
            transformed = self.wct(cf, sf)
            # Interpolate between content and transformed features
            stylized = alpha * transformed + (1 - alpha) * cf
            stylized_features.append(stylized)
        
        # Decode features
        stylized = self.decoder(stylized_features)
        
        # Apply smoothing for photorealism
        stylized = self.smooth_filter(stylized)
        
        return stylized

class PhotoWCTTrainer:
    """
    Trainer class for PhotoWCT
    Handles training and inference
    """
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.model = PhotoWCT().to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config['learning_rate']
        )
        
    def compute_photorealism_loss(self, output, target):
        """Compute loss that encourages photorealism"""
        # Structure similarity loss
        ssim_loss = 1 - F.mse_loss(
            self.model.smooth_filter(output),
            self.model.smooth_filter(target)
        )
        
        # Edge preservation loss
        edge_loss = F.l1_loss(
            F.conv2d(output, self.model.smooth_kernel),
            F.conv2d(target, self.model.smooth_kernel)
        )
        
        return ssim_loss + edge_loss
    
    def train_step(self, content_images, style_images):
        """
        Perform one training step
        
        Args:
            content_images: Batch of content images
            style_images: Batch of style images
            
        Returns:
            dict: Dictionary containing loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Generate stylized images
        stylized = self.model(content_images, style_images)
        
        # Compute losses
        content_loss = F.mse_loss(
            self.model.encoder(stylized)[0],
            self.model.encoder(content_images)[0]
        )
        
        photorealism_loss = self.compute_photorealism_loss(
            stylized, content_images)
        
        # Total loss
        total_loss = (self.config['content_weight'] * content_loss +
                     self.config['photorealism_weight'] * photorealism_loss)
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'content_loss': content_loss.item(),
            'photorealism_loss': photorealism_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def stylize(self, content_image, style_image, alpha=1.0):
        """
        Stylize a single image
        
        Args:
            content_image: Content image tensor
            style_image: Style image tensor
            alpha: Style weight factor
            
        Returns:
            Tensor: Stylized image
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(content_image, style_image, alpha)
