"""
Neural Doodle: Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artwork

Source: https://arxiv.org/abs/1603.01768
Type: CNN (Convolutional Neural Network)

Architecture:
- Semantic map processing
- Patch-based style transfer
- Multi-scale feature matching
- Guided synthesis
- Semantic correspondence

Pros:
- User-guided style transfer
- Semantic control
- Local style control
- Intuitive interface

Cons:
- Requires manual semantic maps
- Computationally intensive
- Sensitive to semantic matching
- Complex parameter tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class PatchMatcher(nn.Module):
    """
    Patch Matcher for Neural Doodles
    Matches patches between content and style images based on semantic maps
    """
    def __init__(self, patch_size=3, stride=1):
        super(PatchMatcher, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        
    def extract_patches(self, x):
        """Extract patches from input tensor"""
        b, c, h, w = x.size()
        patches = F.unfold(x, 
                          kernel_size=self.patch_size,
                          stride=self.stride,
                          padding=self.patch_size//2)
        patches = patches.view(b, c, self.patch_size, self.patch_size, -1)
        patches = patches.permute(0, 4, 1, 2, 3)  # [B, N, C, H, W]
        return patches
    
    def cosine_similarity(self, x, y):
        """Compute cosine similarity between patch sets"""
        x_norm = F.normalize(x.view(x.size(0), x.size(1), -1), dim=2)
        y_norm = F.normalize(y.view(y.size(0), y.size(1), -1), dim=2)
        similarity = torch.bmm(x_norm, y_norm.transpose(1, 2))
        return similarity
    
    def forward(self, content_feat, style_feat, content_mask, style_mask):
        """
        Match patches between content and style features based on semantic masks
        
        Args:
            content_feat: Content feature maps
            style_feat: Style feature maps
            content_mask: Semantic mask for content
            style_mask: Semantic mask for style
            
        Returns:
            Tensor: Matched style patches for reconstruction
        """
        # Extract patches
        content_patches = self.extract_patches(content_feat)
        style_patches = self.extract_patches(style_feat)
        content_mask_patches = self.extract_patches(content_mask)
        style_mask_patches = self.extract_patches(style_mask)
        
        # Compute similarities
        feat_similarity = self.cosine_similarity(content_patches, style_patches)
        mask_similarity = self.cosine_similarity(content_mask_patches, style_mask_patches)
        
        # Combined similarity with mask weighting
        similarity = feat_similarity * mask_similarity
        
        # Find best matching patches
        best_matches = similarity.max(dim=2)[1]
        matched_patches = style_patches[torch.arange(style_patches.size(0)).unsqueeze(1), best_matches]
        
        return matched_patches

class NeuralDoodle(nn.Module):
    """
    Neural Doodle Network
    Implements semantic style transfer with user guidance
    """
    def __init__(self, device='cuda'):
        super(NeuralDoodle, self).__init__()
        self.device = device
        
        # Load and modify VGG16 for feature extraction
        vgg = models.vgg16(pretrained=True).features.eval()
        self.slice1 = nn.Sequential(*list(vgg)[:4])  # relu1_2
        self.slice2 = nn.Sequential(*list(vgg)[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg)[9:16])  # relu3_3
        self.slice4 = nn.Sequential(*list(vgg)[16:23])  # relu4_3
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # Initialize patch matcher
        self.patch_matcher = PatchMatcher()
        
        # Initialize semantic guidance layers
        self.semantic_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.semantic_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.semantic_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.semantic_conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
    def extract_features(self, x):
        """Extract hierarchical features from input"""
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h1, h2, h3, h4
    
    def process_semantic_maps(self, semantic_map):
        """Process semantic maps through guidance layers"""
        s1 = self.semantic_conv1(semantic_map)
        s2 = self.semantic_conv2(F.interpolate(s1, scale_factor=0.5))
        s3 = self.semantic_conv3(F.interpolate(s2, scale_factor=0.5))
        s4 = self.semantic_conv4(F.interpolate(s3, scale_factor=0.5))
        return s1, s2, s3, s4
    
    def reconstruct_feature(self, content_feat, style_feat, content_mask, style_mask):
        """Reconstruct features using patch matching"""
        matched_patches = self.patch_matcher(content_feat, style_feat, content_mask, style_mask)
        return self.patch_matcher.reconstruct_from_patches(matched_patches, content_feat.size())

class NeuralDoodleTrainer:
    """
    Trainer class for Neural Doodle
    Handles training and style transfer with semantic guidance
    """
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.model = NeuralDoodle(self.device).to(self.device)
        
        # Initialize optimizer for generated image
        self.learning_rate = config.get('learning_rate', 1.0)
        
    def compute_loss(self, target_features, current_features, mask_weight=1.0):
        """Compute feature reconstruction loss with semantic guidance"""
        loss = 0
        for target_feat, current_feat in zip(target_features, current_features):
            loss += F.mse_loss(current_feat, target_feat)
        return loss
    
    def style_transfer(self, content_image, style_image, content_mask, style_mask,
                      num_steps=500):
        """
        Perform style transfer with semantic guidance
        
        Args:
            content_image: Content image tensor
            style_image: Style image tensor
            content_mask: Semantic mask for content
            style_mask: Semantic mask for style
            num_steps: Number of optimization steps
            
        Returns:
            Tensor: Stylized image
        """
        # Initialize generated image
        generated = content_image.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([generated], lr=self.learning_rate)
        
        # Extract style features
        style_features = self.model.extract_features(style_image)
        content_features = self.model.extract_features(content_image)
        
        # Process semantic maps
        content_semantic_features = self.model.process_semantic_maps(content_mask)
        style_semantic_features = self.model.process_semantic_maps(style_mask)
        
        for step in range(num_steps):
            def closure():
                optimizer.zero_grad()
                
                # Extract features from generated image
                generated_features = self.model.extract_features(generated)
                
                # Reconstruct features with semantic guidance
                target_features = []
                for gen_feat, style_feat, content_sem, style_sem in zip(
                    generated_features, style_features,
                    content_semantic_features, style_semantic_features):
                    target_feat = self.model.reconstruct_feature(
                        gen_feat, style_feat, content_sem, style_sem)
                    target_features.append(target_feat)
                
                # Compute losses
                content_loss = self.compute_loss(
                    [content_features[-1]], [generated_features[-1]])
                style_loss = self.compute_loss(
                    target_features, generated_features)
                
                # Total loss
                total_loss = (self.config['content_weight'] * content_loss +
                            self.config['style_weight'] * style_loss)
                
                total_loss.backward()
                return total_loss
            
            optimizer.step(closure)
            
            if (step + 1) % 100 == 0:
                print(f'Step [{step+1}/{num_steps}]')
        
        return generated
    
    def transfer_doodle(self, content_image, style_image, doodle, style_regions,
                       num_steps=500):
        """
        Transfer style using doodle guidance
        
        Args:
            content_image: Content image tensor
            style_image: Style image tensor
            doodle: User doodle tensor
            style_regions: Style region masks
            num_steps: Number of optimization steps
            
        Returns:
            Tensor: Stylized image
        """
        # Convert doodle to semantic masks
        content_mask = self.create_semantic_mask(doodle)
        style_mask = self.create_semantic_mask(style_regions)
        
        return self.style_transfer(
            content_image, style_image,
            content_mask, style_mask,
            num_steps=num_steps
        )
    
    @staticmethod
    def create_semantic_mask(doodle, num_colors=10):
        """Convert doodle to one-hot semantic mask"""
        # Quantize colors and create one-hot encoding
        colors = torch.linspace(0, 1, num_colors)
        doodle_flat = doodle.view(-1, 1)
        distances = torch.abs(doodle_flat.unsqueeze(1) - colors)
        color_indices = distances.min(dim=1)[1]
        semantic_mask = F.one_hot(color_indices, num_colors)
        return semantic_mask.view(*doodle.shape[:-1], -1) 