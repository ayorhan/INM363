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
        self.content_weight = config.training.content_weight
        self.style_weight = config.training.style_weight
        self.tv_weight = config.training.tv_weight
        
        # Get layers from model config
        self.content_layers = config.model.content_layers
        self.style_layers = config.model.style_layers
        
        # VGG preprocessing
        self.vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        
        # Load VGG model and move it to GPU
        self.vgg = vgg19(pretrained=True).features.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vgg = self.vgg.to(self.device)
        self.vgg_mean = self.vgg_mean.to(self.device)
        self.vgg_std = self.vgg_std.to(self.device)
        
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
    
    def _preprocess(self, x):
        """Preprocess input for VGG"""
        # Transform from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        # Normalize with VGG mean and std
        x = (x - self.vgg_mean) / self.vgg_std
        return x
    
    def total_variation_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Total variation loss for smoothness"""
        h_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        w_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return h_tv + w_tv
    
    def compute_losses(self, generated, batch):
        # Preprocess images
        generated_prep = self._preprocess(generated)
        content = self._preprocess(batch['content'].to(generated.device))
        style = self._preprocess(batch['style'].to(generated.device))
        
        # Get VGG features
        output_features = {}
        content_features = {}
        style_features = {}
        
        # Extract features for each layer
        x = generated_prep
        x_content = content
        x_style = style
        
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
        
        # Style loss
        style_loss = 0
        for name in [self.layer_mapping[l] for l in self.style_layers]:
            output_gram = self.gram_matrix(output_features[name])
            style_gram = self.gram_matrix(style_features[name])
            style_loss += F.mse_loss(output_gram, style_gram)
        
        # Total variation loss - using the raw generated image
        tv_loss = self.tv_weight * self.total_variation_loss(generated)
        
        return {
            'content': self.content_weight * content_loss,
            'style': self.style_weight * style_loss,
            'tv': tv_loss
        }

class CycleGANLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Get weights from training config
        self.lambda_A = config.training.lambda_A
        self.lambda_B = config.training.lambda_B
        self.lambda_identity = config.training.lambda_identity
        
        # GAN loss type (default to MSE loss)
        self.criterion = nn.MSELoss()
        
    def forward(self, real_A, real_B, fake_A, fake_B, 
                cycle_A, cycle_B, identity_A=None, identity_B=None):
        # Adversarial loss
        loss_G_A = self.criterion(fake_B, torch.ones_like(fake_B))
        loss_G_B = self.criterion(fake_A, torch.ones_like(fake_A))
        
        # Cycle consistency loss
        loss_cycle_A = self.lambda_A * self.criterion(cycle_A, real_A)
        loss_cycle_B = self.lambda_B * self.criterion(cycle_B, real_B)
        
        # Identity loss (optional)
        loss_identity = 0
        if identity_A is not None and identity_B is not None:
            loss_identity = (self.lambda_identity * 
                           (self.criterion(identity_A, real_A) + 
                            self.criterion(identity_B, real_B)))
            
        return {
            'G_A': loss_G_A,
            'G_B': loss_G_B,
            'cycle_A': loss_cycle_A,
            'cycle_B': loss_cycle_B,
            'identity': loss_identity
        }