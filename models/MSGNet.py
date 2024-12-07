"""
MSG-Net: Multi-style Generative Network for Real-time Transfer

Source: https://arxiv.org/abs/1703.06953
Type: CNN (Convolutional Neural Network)

Architecture:
- Multi-style learning
- Inspiration layer
- Style swap module
- Multiple style transfer
- Real-time inference

Pros:
- Fast style transfer
- Multiple style support
- Memory efficient
- Real-time performance

Cons:
- Style capacity limits
- Quality vs speed trade-off
- Fixed style set
- Training complexity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class GramMatrix(nn.Module):
    """
    Compute Gram Matrix for style features
    Used for style loss calculation
    """
    def forward(self, input):
        b, c, h, w = input.size()
        features = input.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)

class InspirationLayer(nn.Module):
    """
    Inspiration Layer for multi-style transfer
    Allows dynamic style switching during inference
    """
    def __init__(self, C, B=1):
        super(InspirationLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, C, C), requires_grad=True)
        self.B = B
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights as identity matrix"""
        self.weight.data.normal_(0, 0.01)
        
    def setTarget(self, target):
        """Set target style features"""
        self.target = target
        
    def forward(self, input):
        # Compute style-transformed features
        self.P = torch.bmm(self.weight.expand(self.B, -1, -1),
                          self.target)
        return torch.bmm(self.P.transpose(1, 2).expand(input.size(0), -1, -1),
                        input.view(input.size(0), input.size(1), -1)).view_as(input)
    
    def __repr__(self):
        return f'InspirationLayer(C={self.weight.size(1)}, B={self.B})'

class ConvBlock(nn.Module):
    """
    Basic convolutional block with instance normalization
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, norm_layer=nn.InstanceNorm2d, activation=nn.ReLU(True)):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding, bias=False)
        self.norm = norm_layer(out_channels) if norm_layer else None
        self.activation = activation
        
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class MSGNet(nn.Module):
    """
    Multi-style Generative Network
    Allows real-time style transfer with multiple styles
    """
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=6, n_styles=32):
        super(MSGNet, self).__init__()
        self.n_styles = n_styles
        
        # Initial convolution blocks
        self.encode1 = ConvBlock(input_nc, ngf, kernel_size=7, padding=3)
        self.encode2 = ConvBlock(ngf, ngf*2, kernel_size=4, stride=2, padding=1)
        self.encode3 = ConvBlock(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1)
        
        # Residual blocks with inspiration layers
        self.res_blocks = nn.ModuleList()
        self.inspiration_layers = nn.ModuleList()
        
        for i in range(n_blocks):
            self.res_blocks.append(nn.Sequential(
                ConvBlock(ngf*4, ngf*4, kernel_size=3, padding=1),
                ConvBlock(ngf*4, ngf*4, kernel_size=3, padding=1, activation=None)
            ))
            self.inspiration_layers.append(InspirationLayer(ngf*4, n_styles))
        
        # Decoder blocks
        self.decode1 = ConvBlock(ngf*4, ngf*2, kernel_size=4, stride=2,
                                padding=1, norm_layer=nn.InstanceNorm2d)
        self.decode2 = ConvBlock(ngf*2, ngf, kernel_size=4, stride=2,
                                padding=1, norm_layer=nn.InstanceNorm2d)
        self.decode3 = ConvBlock(ngf, output_nc, kernel_size=7, padding=3,
                                norm_layer=None, activation=nn.Tanh())
        
        # Style encoder (VGG-based)
        self.style_encoder = StyleEncoder()
        
    def forward(self, x, style_id=0):
        # Encoding
        e1 = self.encode1(x)
        e2 = self.encode2(e1)
        e3 = self.encode3(e2)
        
        # Residual blocks with inspiration
        out = e3
        for res_block, inspiration in zip(self.res_blocks, self.inspiration_layers):
            inspiration.B = self.n_styles
            out = res_block(inspiration(out)) + out
        
        # Decoding
        d1 = self.decode1(out)
        d2 = self.decode2(d1)
        d3 = self.decode3(d2)
        
        return d3
    
    def setTarget(self, target, style_id=0):
        """Set target style features for specific style ID"""
        style_features = self.style_encoder(target)
        for inspiration in self.inspiration_layers:
            inspiration.setTarget(style_features)

class StyleEncoder(nn.Module):
    """
    Style Encoder based on VGG-19
    Extracts style features from style images
    """
    def __init__(self):
        super(StyleEncoder, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg)[:4])  # relu1_1
        self.slice2 = nn.Sequential(*list(vgg)[4:9])  # relu2_1
        self.slice3 = nn.Sequential(*list(vgg)[9:18])  # relu3_1
        self.slice4 = nn.Sequential(*list(vgg)[18:27])  # relu4_1
        
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h4

class MSGNetTrainer:
    """
    Trainer class for MSG-Net
    Handles training and style transfer
    """
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Initialize network
        self.model = MSGNet(
            n_styles=config['n_styles']
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config['lr']
        )
        
        # Initialize loss modules
        self.gram = GramMatrix().to(self.device)
        self.mse_loss = nn.MSELoss()
        
    def train_step(self, content_images, style_images, style_ids):
        """
        Perform one training step
        
        Args:
            content_images: Batch of content images
            style_images: Batch of style images
            style_ids: Style IDs for each image
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Set style targets
        for style_image, style_id in zip(style_images, style_ids):
            self.model.setTarget(style_image.unsqueeze(0), style_id)
        
        # Generate styled images
        output = self.model(content_images, style_ids)
        
        # Calculate content loss
        content_loss = self.mse_loss(
            self.model.style_encoder(output),
            self.model.style_encoder(content_images)
        )
        
        # Calculate style loss
        style_loss = 0
        for style_image, style_id in zip(style_images, style_ids):
            style_features = self.model.style_encoder(style_image.unsqueeze(0))
            output_features = self.model.style_encoder(output)
            
            style_gram = self.gram(style_features)
            output_gram = self.gram(output_features)
            
            style_loss += self.mse_loss(output_gram, style_gram)
        
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
    
    def transfer_style(self, content_image, style_image, style_id=0):
        """
        Perform style transfer
        
        Args:
            content_image: Content image to stylize
            style_image: Style image to extract style from
            style_id: Style ID to use
        """
        self.model.eval()
        with torch.no_grad():
            self.model.setTarget(style_image.unsqueeze(0), style_id)
            return self.model(content_image.unsqueeze(0), style_id) 