"""
Loss functions for style transfer models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import vgg19
from typing import Dict, List, Tuple, Optional

class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss"""
    def __init__(self, 
                 content_layers: List[str] = ['relu4_2'],
                 style_layers: List[str] = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'],
                 content_weight: float = 1.0,
                 style_weight: float = 1.0):
        super().__init__()
        
        self.content_weight = content_weight
        self.style_weight = style_weight
        
        # Load pretrained VGG
        vgg = models.vgg19(pretrained=True).features.eval()
        
        # Layer name mapping
        self.layer_mapping = {
            'relu1_1': '2',  'relu1_2': '4',
            'relu2_1': '7',  'relu2_2': '9',
            'relu3_1': '12', 'relu3_2': '14',
            'relu3_3': '16', 'relu3_4': '18',
            'relu4_1': '21', 'relu4_2': '23',
            'relu4_3': '25', 'relu4_4': '27',
            'relu5_1': '30', 'relu5_2': '32',
            'relu5_3': '34', 'relu5_4': '36'
        }
        
        # Extract required layers
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.layers = nn.ModuleDict()
        
        current_block = nn.Sequential()
        for name, layer in vgg.named_children():
            current_block.add_module(name, layer)
            if name in [self.layer_mapping[l] for l in content_layers + style_layers]:
                self.layers[name] = current_block
                current_block = nn.Sequential()
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix"""
        b, c, h, w = x.size()
        features = x.view(b, c, -1)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        content_loss = 0
        style_loss = 0
        
        for name, layer in self.layers.items():
            input_features = layer(input)
            target_features = layer(target)
            
            # Content loss
            if name in [self.layer_mapping[l] for l in self.content_layers]:
                content_loss += F.mse_loss(input_features, target_features)
            
            # Style loss
            if name in [self.layer_mapping[l] for l in self.style_layers]:
                input_gram = self.gram_matrix(input_features)
                target_gram = self.gram_matrix(target_features)
                style_loss += F.mse_loss(input_gram, target_gram)
        
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        
        return {
            'total': total_loss,
            'content': content_loss.item(),
            'style': style_loss.item()
        }

class AdversarialLoss(nn.Module):
    """Adversarial loss with different modes"""
    def __init__(self, mode: str = 'lsgan'):
        super().__init__()
        self.mode = mode
        if mode == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        elif mode == 'lsgan':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported adversarial loss mode: {mode}")
    
    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.criterion(pred, target)

class CycleLoss(nn.Module):
    """Cycle consistency loss"""
    def __init__(self, lambda_cycle: float = 10.0):
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.criterion = nn.L1Loss()
    
    def forward(self, real: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        return self.lambda_cycle * self.criterion(real, reconstructed)

class IdentityLoss(nn.Module):
    """Identity loss for cycle consistency"""
    def __init__(self, lambda_identity: float = 0.5):
        super().__init__()
        self.lambda_identity = lambda_identity
        self.criterion = nn.L1Loss()
    
    def forward(self, real: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        return self.lambda_identity * self.criterion(real, identity)

class TotalVariationLoss(nn.Module):
    """Total variation loss for smoothness"""
    def __init__(self, weight: float = 1e-6):
        super().__init__()
        self.weight = weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        w_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return self.weight * (h_tv + w_tv)

class StyleTransferLoss(nn.Module):
    """Combined loss for style transfer"""
    def __init__(self, config):
        super().__init__()
        self.content_weight = getattr(config.model, 'content_weight', 1.0)
        self.style_weight = getattr(config.model, 'style_weight', 10.0)
        
        # Load VGG model and move it to GPU
        self.vgg = vgg19(pretrained=True).features.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vgg = self.vgg.to(self.device)
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Layer name mapping
        self.layer_mapping = {
            'relu1_1': '2',  'relu1_2': '4',
            'relu2_1': '7',  'relu2_2': '9',
            'relu3_1': '12', 'relu3_2': '14',
            'relu3_3': '16', 'relu3_4': '18',
            'relu4_1': '21', 'relu4_2': '23',
            'relu4_3': '25', 'relu4_4': '27',
            'relu5_1': '30', 'relu5_2': '32',
            'relu5_3': '34', 'relu5_4': '36'
        }
        
        # Extract required layers
        self.content_layers = ['relu4_2']
        self.style_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        self.layers = nn.ModuleDict()
        
        current_block = nn.Sequential()
        for name, layer in self.vgg.named_children():
            current_block.add_module(name, layer)
            if name in [self.layer_mapping[l] for l in self.content_layers + self.style_layers]:
                self.layers[name] = current_block
                current_block = nn.Sequential()
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for style loss"""
        b, c, h, w = x.size()
        features = x.view(b, c, -1)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def compute_losses(self, outputs: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute content and style losses using VGG features"""
        losses = {}
        
        # Get VGG features
        output_features = {}
        content_features = {}
        style_features = {}
        
        # Extract features for each layer
        x = outputs
        x_content = batch['content']
        x_style = batch['style']
        
        for name, layer in self.layers.items():
            x = layer(x)
            x_content = layer(x_content)
            x_style = layer(x_style)
            
            if name in [self.layer_mapping[l] for l in self.content_layers]:
                output_features[name] = x
                content_features[name] = x_content
                
            if name in [self.layer_mapping[l] for l in self.style_layers]:
                output_features[name] = x
                style_features[name] = x_style
        
        # Content loss
        content_loss = 0
        for name in [self.layer_mapping[l] for l in self.content_layers]:
            content_loss += F.mse_loss(output_features[name], content_features[name])
        losses['content'] = self.content_weight * content_loss
        
        # Style loss
        style_loss = 0
        for name in [self.layer_mapping[l] for l in self.style_layers]:
            output_gram = self.gram_matrix(output_features[name])
            style_gram = self.gram_matrix(style_features[name])
            style_loss += F.mse_loss(output_gram, style_gram)
        losses['style'] = self.style_weight * style_loss
        
        return losses 