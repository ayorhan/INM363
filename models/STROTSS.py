"""
STROTSS: Style Transfer by Relaxed Optimal Transport and Self-Similarity

Source: https://arxiv.org/abs/1904.12785
Type: CNN (Convolutional Neural Network)

Architecture:
- Optimal transport matching
- Self-similarity preservation
- Multi-scale feature extraction
- Relaxed optimization
- Content-style balancing

Pros:
- High-quality results
- Better content preservation
- Flexible style transfer
- Robust to different styles

Cons:
- Slow optimization process
- High memory usage
- Complex OT computation
- Parameter sensitivity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from scipy.optimize import linear_sum_assignment

class STROTSS(nn.Module):
    """
    Style Transfer by Relaxed Optimal Transport and Self-Similarity
    """
    def __init__(self, device='cuda'):
        super(STROTSS, self).__init__()
        self.device = device
        
        # Initialize VGG for feature extraction
        self.vgg = VGGFeatures().to(device)
        self.vgg.eval()
        
        # Feature layers for style and content
        self.style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']
        self.content_layers = ['relu4_1']
        
        # Hyperparameters
        self.style_weight = 1.0
        self.content_weight = 1.0
        self.ss_weight = 1.0  # Self-similarity weight
        
    def extract_features(self, x, layers):
        """Extract VGG features at specified layers"""
        features = {}
        for name, feat in self.vgg(x, layers):
            features[name] = feat
        return features

    def compute_style_cost(self, source_features, target_features, layer):
        """
        Compute style cost using relaxed optimal transport
        """
        # Reshape features
        source_feat = source_features[layer].view(source_features[layer].size(0), -1)
        target_feat = target_features[layer].view(target_features[layer].size(0), -1)
        
        # Normalize features
        source_feat = source_feat / torch.norm(source_feat, dim=1, keepdim=True)
        target_feat = target_feat / torch.norm(target_feat, dim=1, keepdim=True)
        
        # Compute cost matrix
        cost_matrix = 1 - torch.mm(source_feat, target_feat.t())
        
        # Solve optimal transport (Sinkhorn algorithm)
        transport_plan = self.sinkhorn(cost_matrix)
        
        return torch.sum(transport_plan * cost_matrix)

    def sinkhorn(self, cost_matrix, epsilon=0.1, num_iters=50):
        """
        Sinkhorn algorithm for optimal transport
        """
        n, m = cost_matrix.size()
        
        # Initialize transport plan
        log_mu = torch.zeros(n, 1).to(self.device)
        log_nu = torch.zeros(m, 1).to(self.device)
        
        # Sinkhorn iterations
        u = torch.zeros(n, 1).to(self.device)
        for _ in range(num_iters):
            v = log_nu - torch.logsumexp(
                -cost_matrix/epsilon + u, dim=0, keepdim=True).t()
            u = log_mu - torch.logsumexp(
                -cost_matrix/epsilon + v.t(), dim=1, keepdim=True)
        
        # Compute transport plan
        transport_plan = torch.exp(-cost_matrix/epsilon + u + v.t())
        return transport_plan

    def compute_self_similarity(self, features):
        """
        Compute self-similarity matrix
        """
        # Reshape features
        feat = features.view(features.size(0), -1)
        
        # Normalize features
        feat = feat / torch.norm(feat, dim=1, keepdim=True)
        
        # Compute self-similarity
        similarity = torch.mm(feat, feat.t())
        
        return similarity

    def compute_content_loss(self, source_features, target_features, layer):
        """
        Compute content loss using feature matching
        """
        return F.mse_loss(
            source_features[layer],
            target_features[layer]
        )

    def style_transfer(self, content_image, style_image, num_steps=500):
        """
        Perform style transfer optimization
        
        Args:
            content_image: Content image tensor
            style_image: Style image tensor
            num_steps: Number of optimization steps
            
        Returns:
            Tensor: Stylized image
        """
        # Initialize output image
        output = content_image.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([output])
        
        # Extract style features
        style_features = self.extract_features(style_image, self.style_layers)
        
        # Compute style self-similarity
        style_ss = {
            layer: self.compute_self_similarity(style_features[layer])
            for layer in self.style_layers
        }
        
        # Optimization loop
        for step in range(num_steps):
            def closure():
                optimizer.zero_grad()
                
                # Extract features from current output
                output_features = self.extract_features(
                    output, self.style_layers + self.content_layers)
                
                # Content loss
                content_loss = 0
                for layer in self.content_layers:
                    content_features = self.extract_features(
                        content_image, [layer])[layer]
                    content_loss += self.compute_content_loss(
                        output_features, {'relu4_1': content_features}, layer)
                
                # Style loss
                style_loss = 0
                for layer in self.style_layers:
                    style_loss += self.compute_style_cost(
                        output_features, style_features, layer)
                
                # Self-similarity loss
                ss_loss = 0
                for layer in self.style_layers:
                    output_ss = self.compute_self_similarity(
                        output_features[layer])
                    ss_loss += F.mse_loss(output_ss, style_ss[layer])
                
                # Total loss
                total_loss = (self.content_weight * content_loss +
                            self.style_weight * style_loss +
                            self.ss_weight * ss_loss)
                
                total_loss.backward()
                return total_loss
            
            optimizer.step(closure)
        
        return output

class VGGFeatures(nn.Module):
    """
    VGG feature extractor with named layers
    """
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        
        self.layers = nn.ModuleList([])
        self.layer_names = []
        
        # Create sequential blocks
        current_block = []
        block_count = 1
        relu_count = 1
        
        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):
                current_block.append(layer)
            elif isinstance(layer, nn.ReLU):
                current_block.append(nn.ReLU(inplace=False))
                self.layers.append(nn.Sequential(*current_block))
                self.layer_names.append(f'relu{block_count}_{relu_count}')
                current_block = []
                relu_count += 1
            elif isinstance(layer, nn.MaxPool2d):
                current_block.append(layer)
                block_count += 1
                relu_count = 1
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x, layers=None):
        """
        Forward pass with specified layers
        
        Args:
            x: Input tensor
            layers: List of layer names to extract
            
        Returns:
            List of (name, feature) tuples
        """
        if layers is None:
            layers = self.layer_names
            
        features = []
        for name, layer in zip(self.layer_names, self.layers):
            x = layer(x)
            if name in layers:
                features.append((name, x))
                
        return features

class STROTSSModel:
    """
    Complete STROTSS model with training and inference
    """
    def __init__(self, config):
        self.device = config['device']
        self.config = config
        
        # Initialize STROTSS
        self.model = STROTSS(device=self.device)
        
        # Set hyperparameters
        self.model.style_weight = config.get('style_weight', 1.0)
        self.model.content_weight = config.get('content_weight', 1.0)
        self.model.ss_weight = config.get('ss_weight', 1.0)
        
    def style_transfer(self, content_image, style_image, num_steps=500):
        """
        Perform style transfer
        
        Args:
            content_image: Content image tensor
            style_image: Style image tensor
            num_steps: Number of optimization steps
            
        Returns:
            Tensor: Stylized image
        """
        return self.model.style_transfer(
            content_image, style_image, num_steps)
    
    def compute_loss_metrics(self, output, content_image, style_image):
        """
        Compute detailed loss metrics
        
        Args:
            output: Output image tensor
            content_image: Content image tensor
            style_image: Style image tensor
            
        Returns:
            dict: Dictionary of loss metrics
        """
        # Extract features
        output_features = self.model.extract_features(
            output, self.model.style_layers + self.model.content_layers)
        style_features = self.model.extract_features(
            style_image, self.model.style_layers)
        content_features = self.model.extract_features(
            content_image, self.model.content_layers)
        
        # Compute losses
        content_loss = sum(
            self.model.compute_content_loss(
                output_features, {'relu4_1': content_features[layer]}, layer)
            for layer in self.model.content_layers
        )
        
        style_loss = sum(
            self.model.compute_style_cost(
                output_features, style_features, layer)
            for layer in self.model.style_layers
        )
        
        # Compute self-similarity losses
        style_ss = {
            layer: self.model.compute_self_similarity(style_features[layer])
            for layer in self.model.style_layers
        }
        
        ss_loss = sum(
            F.mse_loss(
                self.model.compute_self_similarity(output_features[layer]),
                style_ss[layer]
            )
            for layer in self.model.style_layers
        )
        
        return {
            'content_loss': content_loss.item(),
            'style_loss': style_loss.item(),
            'self_similarity_loss': ss_loss.item(),
            'total_loss': (content_loss + style_loss + ss_loss).item()
        } 