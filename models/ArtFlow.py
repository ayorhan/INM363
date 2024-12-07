"""
ArtFlow: Unbiased Image Style Transfer via Reversible Neural Flows

Source: https://arxiv.org/abs/2103.16877
Type: Flow-based Neural Network

Architecture:
- Reversible feature extraction
- Normalizing flows for style transfer
- Bijective transformations
- Multi-scale architecture
- Invertible neural networks

Pros:
- Theoretically sound approach
- Reversible transformations
- Unbiased style transfer
- Good style preservation

Cons:
- Complex implementation
- Slower than traditional methods
- Limited by flow architecture
- Training instability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class FlowEncoder(nn.Module):
    """
    Encoder with normalizing flow components
    """
    def __init__(self):
        super(FlowEncoder, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        
        # Create encoder blocks
        self.block1 = nn.Sequential(*list(vgg)[:4])   # relu1_1
        self.block2 = nn.Sequential(*list(vgg)[4:9])  # relu2_1
        self.block3 = nn.Sequential(*list(vgg)[9:18]) # relu3_1
        self.block4 = nn.Sequential(*list(vgg)[18:27])# relu4_1
        
        # Flow modules
        self.flow1 = NormalizingFlowBlock(64)
        self.flow2 = NormalizingFlowBlock(128)
        self.flow3 = NormalizingFlowBlock(256)
        self.flow4 = NormalizingFlowBlock(512)
        
        # Freeze VGG parameters
        for param in self.parameters():
            if not isinstance(param, nn.Parameter):
                param.requires_grad = False
                
    def forward(self, x, reverse=False):
        if not reverse:
            # Forward flow
            feat1, log_det1 = self.flow1(self.block1(x))
            feat2, log_det2 = self.flow2(self.block2(feat1))
            feat3, log_det3 = self.flow3(self.block3(feat2))
            feat4, log_det4 = self.flow4(self.block4(feat3))
            
            log_det = log_det1 + log_det2 + log_det3 + log_det4
            return [feat1, feat2, feat3, feat4], log_det
        else:
            # Reverse flow
            feat4 = self.flow4.reverse(x[3])
            feat3 = self.flow3.reverse(self.block4(feat4))
            feat2 = self.flow2.reverse(self.block3(feat3))
            feat1 = self.flow1.reverse(self.block2(feat2))
            return [feat1, feat2, feat3, feat4]

class NormalizingFlowBlock(nn.Module):
    """
    Normalizing flow block for invertible feature transformation
    """
    def __init__(self, channels):
        super(NormalizingFlowBlock, self).__init__()
        
        # Coupling layers
        self.coupling1 = AffineCouplingLayer(channels)
        self.coupling2 = AffineCouplingLayer(channels)
        self.coupling3 = AffineCouplingLayer(channels)
        
        # 1x1 Convolution for channel mixing
        self.conv1x1 = Invertible1x1Conv(channels)
        
        # Activation normalization
        self.actnorm = ActNorm(channels)
        
    def forward(self, x):
        total_log_det = 0
        
        # Forward flow
        x, log_det = self.actnorm(x)
        total_log_det += log_det
        
        x, log_det = self.conv1x1(x)
        total_log_det += log_det
        
        x, log_det = self.coupling1(x)
        total_log_det += log_det
        
        x, log_det = self.coupling2(x)
        total_log_det += log_det
        
        x, log_det = self.coupling3(x)
        total_log_det += log_det
        
        return x, total_log_det
    
    def reverse(self, x):
        x = self.coupling3.reverse(x)
        x = self.coupling2.reverse(x)
        x = self.coupling1.reverse(x)
        x = self.conv1x1.reverse(x)
        x = self.actnorm.reverse(x)
        return x

class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer for normalizing flows
    """
    def __init__(self, channels):
        super(AffineCouplingLayer, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(channels//2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )
        
    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        
        h = self.net(x1)
        shift, scale = torch.chunk(h, 2, dim=1)
        scale = torch.sigmoid(scale + 2.0)
        
        y2 = x2 * scale + shift
        y = torch.cat([x1, y2], dim=1)
        
        log_det = torch.sum(torch.log(scale), dim=[1, 2, 3])
        return y, log_det
    
    def reverse(self, y):
        y1, y2 = torch.chunk(y, 2, dim=1)
        
        h = self.net(y1)
        shift, scale = torch.chunk(h, 2, dim=1)
        scale = torch.sigmoid(scale + 2.0)
        
        x2 = (y2 - shift) / scale
        x = torch.cat([y1, x2], dim=1)
        return x

class Invertible1x1Conv(nn.Module):
    """
    Invertible 1x1 convolution for channel mixing
    """
    def __init__(self, channels):
        super(Invertible1x1Conv, self).__init__()
        
        w_init = torch.qr(torch.randn(channels, channels))[0]
        self.weight = nn.Parameter(w_init)
        
    def forward(self, x):
        b, c, h, w = x.size()
        weight = self.weight.view(c, c, 1, 1)
        
        z = F.conv2d(x, weight)
        log_det = h * w * torch.slogdet(self.weight)[1]
        
        return z, log_det
    
    def reverse(self, z):
        b, c, h, w = z.size()
        weight = self.weight.inverse().view(c, c, 1, 1)
        x = F.conv2d(z, weight)
        return x

class ActNorm(nn.Module):
    """
    Activation normalization layer
    """
    def __init__(self, channels):
        super(ActNorm, self).__init__()
        
        self.loc = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.initialized = False
        
    def initialize(self, x):
        with torch.no_grad():
            mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
            std = torch.std(x, dim=[0, 2, 3], keepdim=True)
            
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))
            
    def forward(self, x):
        if not self.initialized:
            self.initialize(x)
            self.initialized = True
            
        z = self.scale * (x + self.loc)
        log_det = torch.sum(torch.log(torch.abs(self.scale)), dim=[1, 2, 3])
        
        return z, log_det
    
    def reverse(self, z):
        x = z / self.scale - self.loc
        return x

class ArtFlow:
    """
    Complete ArtFlow model for style transfer
    """
    def __init__(self, config):
        self.device = config['device']
        self.config = config
        
        # Initialize networks
        self.encoder = FlowEncoder().to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=config['lr']
        )
        
    def style_transfer(self, content_image, style_image):
        """
        Perform style transfer using normalizing flows
        
        Args:
            content_image: Content image tensor
            style_image: Style image tensor
            
        Returns:
            Tensor: Stylized image
        """
        # Forward flow
        content_features, c_log_det = self.encoder(content_image)
        style_features, s_log_det = self.encoder(style_image)
        
        # Mix features in latent space
        mixed_features = self.mix_features(content_features, style_features)
        
        # Reverse flow
        stylized = self.encoder(mixed_features, reverse=True)
        return stylized
    
    def mix_features(self, content_features, style_features):
        """Mix content and style features in latent space"""
        mixed = []
        for cf, sf in zip(content_features, style_features):
            # Adaptive mixing based on feature statistics
            c_mean, c_std = self.calc_mean_std(cf)
            s_mean, s_std = self.calc_mean_std(sf)
            
            normalized = (cf - c_mean) / c_std
            mixed_feat = normalized * s_std + s_mean
            mixed.append(mixed_feat)
        
        return mixed
    
    def calc_mean_std(self, feat):
        """Calculate mean and standard deviation of features"""
        mean = feat.mean([2, 3], keepdim=True)
        std = feat.std([2, 3], keepdim=True) + 1e-6
        return mean, std
    
    def compute_loss(self, stylized, content_image, style_image):
        """Compute style transfer losses"""
        # Extract features
        stylized_features, _ = self.encoder(stylized)
        content_features, _ = self.encoder(content_image)
        style_features, _ = self.encoder(style_image)
        
        # Content loss
        content_loss = F.mse_loss(stylized_features[-1], content_features[-1])
        
        # Style loss with gram matrices
        style_loss = 0
        for sf, stf in zip(style_features, stylized_features):
            sf_gram = self.gram_matrix(sf)
            stf_gram = self.gram_matrix(stf)
            style_loss += F.mse_loss(stf_gram, sf_gram)
        
        return content_loss, style_loss
    
    def gram_matrix(self, x):
        """Compute gram matrix"""
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
        
        # Compute losses
        content_loss, style_loss = self.compute_loss(
            stylized, content_images, style_images)
        
        # Flow loss
        _, flow_loss = self.encoder(stylized)
        
        # Total loss
        total_loss = (self.config['content_weight'] * content_loss +
                     self.config['style_weight'] * style_loss +
                     self.config['flow_weight'] * flow_loss)
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'content_loss': content_loss.item(),
            'style_loss': style_loss.item(),
            'flow_loss': flow_loss.item(),
            'total_loss': total_loss.item()
        } 