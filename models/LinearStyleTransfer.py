"""
Linear Style Transfer: Efficient Neural Style Transfer with Linear Transformations

Source: Inspired by https://arxiv.org/abs/1705.08086
Type: CNN (Convolutional Neural Network)

Architecture:
- Linear transformation-based encoder
- Feature transformation through 1x1 convolutions
- Efficient decoder with skip connections
- Gram matrix style representation
- Multi-level feature processing

Pros:
- Fast inference time
- Memory efficient
- Good style transfer quality
- Stable training process

Cons:
- Less expressive than non-linear methods
- May miss complex style patterns
- Limited by linear transformations
- Requires careful feature selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class LinearEncoder(nn.Module):
    """
    Encoder network with linear transformation layers
    Designed for efficient style transfer
    """
    def __init__(self):
        super(LinearEncoder, self).__init__()
        # Load pretrained VGG-19 features
        vgg = models.vgg19(pretrained=True).features
        
        # Create encoder blocks
        self.block1 = nn.Sequential(*list(vgg)[:4])  # relu1_1
        self.block2 = nn.Sequential(*list(vgg)[4:9])  # relu2_1
        self.block3 = nn.Sequential(*list(vgg)[9:18])  # relu3_1
        self.block4 = nn.Sequential(*list(vgg)[18:27])  # relu4_1
        
        # Linear transformation layers
        self.linear1 = nn.Conv1d(64, 64, 1)
        self.linear2 = nn.Conv1d(128, 128, 1)
        self.linear3 = nn.Conv1d(256, 256, 1)
        self.linear4 = nn.Conv1d(512, 512, 1)
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Extract features
        feat1 = self.block1(x)
        feat2 = self.block2(feat1)
        feat3 = self.block3(feat2)
        feat4 = self.block4(feat3)
        
        # Apply linear transformations
        b, c, h, w = feat1.size()
        feat1_flat = feat1.view(b, c, -1)
        feat1_linear = self.linear1(feat1_flat).view(b, c, h, w)
        
        b, c, h, w = feat2.size()
        feat2_flat = feat2.view(b, c, -1)
        feat2_linear = self.linear2(feat2_flat).view(b, c, h, w)
        
        b, c, h, w = feat3.size()
        feat3_flat = feat3.view(b, c, -1)
        feat3_linear = self.linear3(feat3_flat).view(b, c, h, w)
        
        b, c, h, w = feat4.size()
        feat4_flat = feat4.view(b, c, -1)
        feat4_linear = self.linear4(feat4_flat).view(b, c, h, w)
        
        return [feat1_linear, feat2_linear, feat3_linear, feat4_linear]

class LinearDecoder(nn.Module):
    """
    Decoder network with linear transformation layers
    Efficiently reconstructs images from encoded features
    """
    def __init__(self):
        super(LinearDecoder, self).__init__()
        
        # Linear transformation layers
        self.linear4 = nn.Conv1d(512, 512, 1)
        self.linear3 = nn.Conv1d(256, 256, 1)
        self.linear2 = nn.Conv1d(128, 128, 1)
        self.linear1 = nn.Conv1d(64, 64, 1)
        
        # Decoder layers
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, features):
        # Apply linear transformations and decode
        feat4, feat3, feat2, feat1 = features
        
        # Linear transformations
        b, c, h, w = feat4.size()
        feat4_flat = feat4.view(b, c, -1)
        feat4_linear = self.linear4(feat4_flat).view(b, c, h, w)
        
        x = self.deconv4(feat4_linear)
        
        b, c, h, w = feat3.size()
        feat3_flat = feat3.view(b, c, -1)
        feat3_linear = self.linear3(feat3_flat).view(b, c, h, w)
        x = x + feat3_linear
        x = self.deconv3(x)
        
        b, c, h, w = feat2.size()
        feat2_flat = feat2.view(b, c, -1)
        feat2_linear = self.linear2(feat2_flat).view(b, c, h, w)
        x = x + feat2_linear
        x = self.deconv2(x)
        
        b, c, h, w = feat1.size()
        feat1_flat = feat1.view(b, c, -1)
        feat1_linear = self.linear1(feat1_flat).view(b, c, h, w)
        x = x + feat1_linear
        x = self.deconv1(x)
        
        return x

class LinearStyleTransfer(nn.Module):
    """
    Linear Style Transfer model
    Combines encoder and decoder with linear transformations
    """
    def __init__(self):
        super(LinearStyleTransfer, self).__init__()
        self.encoder = LinearEncoder()
        self.decoder = LinearDecoder()
        
    def compute_gram_matrix(self, x):
        """Compute Gram matrix for style loss"""
        b, c, h, w = x.size()
        features = x.view(b, c, -1)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)
    
    def forward(self, content, style, alpha=1.0):
        """
        Forward pass for style transfer
        
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
        
        # Compute style statistics
        style_grams = [self.compute_gram_matrix(feat) for feat in style_features]
        
        # Linear interpolation of features
        transferred_features = []
        for cf, sf, gram in zip(content_features, style_features, style_grams):
            # Compute content feature statistics
            b, c, h, w = cf.size()
            cf_flat = cf.view(b, c, -1)
            cf_gram = torch.bmm(cf_flat, cf_flat.transpose(1, 2)).div(c * h * w)
            
            # Linear interpolation
            transferred = cf + alpha * (torch.bmm(gram, cf_flat).view(b, c, h, w) - cf)
            transferred_features.append(transferred)
        
        # Decode features
        stylized = self.decoder(transferred_features)
        return stylized

class LinearStyleTransferTrainer:
    """
    Trainer class for Linear Style Transfer
    Handles training and inference
    """
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.model = LinearStyleTransfer().to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config['learning_rate']
        )
        
    def compute_content_loss(self, input_features, target_features):
        """Compute content loss"""
        return sum(F.mse_loss(input_feat, target_feat) 
                  for input_feat, target_feat in zip(input_features, target_features))
    
    def compute_style_loss(self, input_features, target_features):
        """Compute style loss using Gram matrices"""
        loss = 0
        for input_feat, target_feat in zip(input_features, target_features):
            input_gram = self.model.compute_gram_matrix(input_feat)
            target_gram = self.model.compute_gram_matrix(target_feat)
            loss += F.mse_loss(input_gram, target_gram)
        return loss
    
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
        
        # Extract features for loss computation
        content_features = self.model.encoder(content_images)
        style_features = self.model.encoder(style_images)
        stylized_features = self.model.encoder(stylized)
        
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