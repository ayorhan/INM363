"""
Neural Style Transfer (Gatys Method)

Source: https://arxiv.org/abs/1508.06576
Type: CNN (Convolutional Neural Network)

Architecture:
- VGG19-based feature extraction
- Gram matrix style representation
- Multi-level feature matching
- Optimization-based transfer
- Content and style loss balancing

Pros:
- Original neural style transfer method
- Flexible style adaptation
- No training required
- Works with any style image

Cons:
- Slow optimization process
- High memory usage
- Inconsistent results
- Limited by VGG features
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class GatysModel(nn.Module):
    """
    Implementation of "A Neural Algorithm of Artistic Style" (Gatys et al., 2015)
    Uses VGG19 for feature extraction and optimizes directly on the image.
    """
    def __init__(self, content_layers=['conv4_2'], style_layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']):
        super(GatysModel, self).__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers
        
        # Load pretrained VGG19 and set to eval mode
        vgg = models.vgg19(pretrained=True).features.eval()
        self.vgg = vgg
        
        # Create layer mapping
        self.layer_mapping = {
            'conv1_1': '0', 'conv1_2': '2',
            'conv2_1': '5', 'conv2_2': '7',
            'conv3_1': '10', 'conv3_2': '12', 'conv3_3': '14', 'conv3_4': '16',
            'conv4_1': '19', 'conv4_2': '21', 'conv4_3': '23', 'conv4_4': '25',
            'conv5_1': '28', 'conv5_2': '30', 'conv5_3': '32', 'conv5_4': '34'
        }
        
        # Freeze parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
            
    def _get_features(self, x, layers):
        """Extract features from specified layers
        
        Args:
            x (torch.Tensor): Input image tensor
            layers (list): List of layer names to extract features from
            
        Returns:
            dict: Features from specified layers
        """
        features = {}
        current = x
        for name, module in self.vgg._modules.items():
            current = module(current)
            for layer in layers:
                if name == self.layer_mapping[layer]:
                    features[layer] = current
        return features
    
    def forward(self, content_img, style_img, num_steps=300, style_weight=1e6, content_weight=1):
        """Forward pass implementing Gatys style transfer
        
        Args:
            content_img (torch.Tensor): Content image tensor
            style_img (torch.Tensor): Style image tensor
            num_steps (int): Number of optimization steps
            style_weight (float): Weight for style loss
            content_weight (float): Weight for content loss
            
        Returns:
            torch.Tensor: Stylized image tensor
        """
        # Initialize generated image with content image
        generated = content_img.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([generated])
        
        # Extract style and content features
        content_features = self._get_features(content_img, self.content_layers)
        style_features = self._get_features(style_img, self.style_layers)
        
        # Compute gram matrices for style features
        style_grams = {
            layer: self._gram_matrix(style_features[layer])
            for layer in self.style_layers
        }
        
        def closure():
            optimizer.zero_grad()
            
            # Get features of generated image
            generated_features = self._get_features(generated, self.content_layers + self.style_layers)
            
            # Content loss
            content_loss = 0
            for layer in self.content_layers:
                generated_feat = generated_features[layer]
                content_feat = content_features[layer].detach()
                content_loss += F.mse_loss(generated_feat, content_feat)
            
            # Style loss
            style_loss = 0
            for layer in self.style_layers:
                generated_feat = generated_features[layer]
                generated_gram = self._gram_matrix(generated_feat)
                style_gram = style_grams[layer].detach()
                style_loss += F.mse_loss(generated_gram, style_gram)
            
            # Total loss
            total_loss = content_weight * content_loss + style_weight * style_loss
            total_loss.backward()
            
            return total_loss
        
        # Optimization loop
        for _ in range(num_steps):
            optimizer.step(closure)
        
        return generated
    
    def _gram_matrix(self, x):
        """Compute Gram matrix for style loss
        
        Args:
            x (torch.Tensor): Feature tensor
            
        Returns:
            torch.Tensor: Gram matrix
        """
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)

def content_loss(gen_features, content_features):
    """Compute content loss between generated and content features
    
    Args:
        gen_features (torch.Tensor): Generated image features
        content_features (torch.Tensor): Content image features
        
    Returns:
        torch.Tensor: Content loss
    """
    return F.mse_loss(gen_features, content_features)

def style_loss(gen_features, style_features):
    """Compute style loss using Gram matrices
    
    Args:
        gen_features (torch.Tensor): Generated image features
        style_features (torch.Tensor): Style image features
        
    Returns:
        torch.Tensor: Style loss
    """
    # Compute Gram matrices
    gen_gram = GatysModel._gram_matrix(None, gen_features)
    style_gram = GatysModel._gram_matrix(None, style_features)
    return F.mse_loss(gen_gram, style_gram) 