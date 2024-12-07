"""
AdaIN: Adaptive Instance Normalization for Style Transfer

Source: https://arxiv.org/abs/1703.06868
Type: CNN (Convolutional Neural Network)

Architecture:
- VGG-based encoder
- Adaptive instance normalization layer
- Decoder with upsampling
- Style-content alignment
- Feature statistics matching

Pros:
- Fast arbitrary style transfer
- Lightweight architecture
- Good style adaptation
- Real-time performance

Cons:
- Can lose fine details
- Limited by instance normalization
- May produce artifacts
- Style strength sensitivity
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder using VGG19 layers up to relu4_1
    Used to extract content and style features
    """
    def __init__(self):
        super(Encoder, self).__init__()
        # Load pretrained VGG19 and slice until relu4_1
        vgg = models.vgg19(pretrained=True).features[:21]
        
        # Create sequential model with first 4 blocks of VGG
        self.slice1 = nn.Sequential(*list(vgg)[:4])  # relu1_1
        self.slice2 = nn.Sequential(*list(vgg)[4:9])  # relu2_1
        self.slice3 = nn.Sequential(*list(vgg)[9:18])  # relu3_1
        self.slice4 = nn.Sequential(*list(vgg)[18:21])  # relu4_1
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x, encode_only=False):
        """
        Forward pass through encoder
        
        Args:
            x (torch.Tensor): Input tensor
            encode_only (bool): If True, return all features; if False, return only final features
            
        Returns:
            torch.Tensor or list: Features from different layers
        """
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        
        if encode_only:
            return [h1, h2, h3, h4]
        return h4

class Decoder(nn.Module):
    """
    Decoder network to reconstruct image from encoded features
    Mirror structure of the encoder
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        """
        Forward pass through decoder
        
        Args:
            x (torch.Tensor): Encoded features
            
        Returns:
            torch.Tensor: Reconstructed image
        """
        return self.decoder(x)

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization for Style Transfer
    As described in 'Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization'
    """
    def __init__(self):
        super(AdaIN, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def adaptive_instance_normalization(self, content_feat, style_feat):
        """
        Performs adaptive instance normalization
        
        Args:
            content_feat (torch.Tensor): Content features
            style_feat (torch.Tensor): Style features
            
        Returns:
            torch.Tensor: Normalized and adapted features
        """
        size = content_feat.size()
        
        # Calculate mean and standard deviation
        content_mean = torch.mean(content_feat, dim=[2, 3], keepdim=True)
        content_std = torch.std(content_feat, dim=[2, 3], keepdim=True) + 1e-6
        style_mean = torch.mean(style_feat, dim=[2, 3], keepdim=True)
        style_std = torch.std(style_feat, dim=[2, 3], keepdim=True) + 1e-6
        
        # Normalize content features
        normalized_feat = (content_feat - content_mean) / content_std
        
        # Adapt to style statistics
        return normalized_feat * style_std + style_mean
    
    def forward(self, content, style, alpha=1.0):
        """
        Forward pass for style transfer
        
        Args:
            content (torch.Tensor): Content image
            style (torch.Tensor): Style image
            alpha (float): Weight for style interpolation (1.0 = full style transfer)
            
        Returns:
            torch.Tensor: Stylized image
        """
        assert 0 <= alpha <= 1, 'Alpha must be between 0 and 1'
        
        # Extract features
        content_feat = self.encoder(content)
        style_feat = self.encoder(style)
        
        # Perform AdaIN
        t = self.adaptive_instance_normalization(content_feat, style_feat)
        
        # Interpolate between content and stylized features
        t = alpha * t + (1 - alpha) * content_feat
        
        # Decode
        stylized = self.decoder(t)
        return stylized
    
    def calc_content_loss(self, input, target):
        """
        Calculate content loss
        
        Args:
            input (torch.Tensor): Input features
            target (torch.Tensor): Target features
            
        Returns:
            torch.Tensor: Content loss
        """
        return F.mse_loss(input, target)
    
    def calc_style_loss(self, input, target):
        """
        Calculate style loss using mean and standard deviation
        
        Args:
            input (torch.Tensor): Input features
            target (torch.Tensor): Target features
            
        Returns:
            torch.Tensor: Style loss
        """
        input_mean = torch.mean(input, dim=[2, 3])
        input_std = torch.std(input, dim=[2, 3]) + 1e-6
        target_mean = torch.mean(target, dim=[2, 3])
        target_std = torch.std(target, dim=[2, 3]) + 1e-6
        
        return F.mse_loss(input_mean, target_mean) + F.mse_loss(input_std, target_std)

class AdaINStyleTransfer:
    """
    Wrapper class for training and using AdaIN style transfer
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = AdaIN().to(device)
        self.optimizer = torch.optim.Adam(self.model.decoder.parameters(), lr=1e-4)
        
    def train_step(self, content_images, style_images, content_weight=1.0, style_weight=10.0):
        """
        Single training step
        
        Args:
            content_images (torch.Tensor): Batch of content images
            style_images (torch.Tensor): Batch of style images
            content_weight (float): Weight for content loss
            style_weight (float): Weight for style loss
            
        Returns:
            dict: Dictionary containing losses
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        stylized = self.model(content_images, style_images)
        
        # Calculate losses
        content_loss = self.model.calc_content_loss(
            self.model.encoder(stylized),
            self.model.encoder(content_images)
        )
        
        style_feats = self.model.encoder(style_images, encode_only=True)
        stylized_feats = self.model.encoder(stylized, encode_only=True)
        
        style_loss = 0
        for s_feat, st_feat in zip(style_feats, stylized_feats):
            style_loss += self.model.calc_style_loss(st_feat, s_feat)
        
        # Total loss
        loss = content_weight * content_loss + style_weight * style_loss
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return {
            'content_loss': content_loss.item(),
            'style_loss': style_loss.item(),
            'total_loss': loss.item()
        }
    
    def stylize(self, content_image, style_image, alpha=1.0):
        """
        Stylize a single image
        
        Args:
            content_image (torch.Tensor): Content image
            style_image (torch.Tensor): Style image
            alpha (float): Style interpolation weight
            
        Returns:
            torch.Tensor: Stylized image
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(content_image, style_image, alpha) 