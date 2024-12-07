"""
Deep Image Analogy: Finding Visual Analogies through Deep Features

Source: https://arxiv.org/abs/1705.01088
Type: CNN (Convolutional Neural Network)

Architecture:
- VGG-based feature extractor
- PatchMatch algorithm for dense correspondence
- Bidirectional similarity computation
- Multi-level feature reconstruction
- Edge-aware refinement

Pros:
- Creates semantically meaningful analogies
- Handles complex visual relationships
- Good detail preservation
- Robust to style variations

Cons:
- Computationally intensive PatchMatch
- Memory intensive for high-resolution images
- Sensitive to feature selection
- May struggle with large appearance changes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class FeatureExtractor(nn.Module):
    """
    VGG-based feature extractor for deep image analogy
    Extracts hierarchical features with additional processing
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg = models.vgg19(pretrained=True).features.eval()
        
        # Create feature extraction layers
        self.slice1 = nn.Sequential(*list(vgg)[:4])    # relu1_2
        self.slice2 = nn.Sequential(*list(vgg)[4:9])   # relu2_2
        self.slice3 = nn.Sequential(*list(vgg)[9:18])  # relu3_4
        self.slice4 = nn.Sequential(*list(vgg)[18:27]) # relu4_4
        self.slice5 = nn.Sequential(*list(vgg)[27:36]) # relu5_4
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x, extract_list=None):
        """
        Extract features at multiple levels
        
        Args:
            x: Input image tensor
            extract_list: List of layer indices to extract features from
            
        Returns:
            list: List of feature maps at specified layers
        """
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        
        if extract_list is None:
            return [h1, h2, h3, h4, h5]
        else:
            out = []
            for i in extract_list:
                if i == 0: out.append(h1)
                elif i == 1: out.append(h2)
                elif i == 2: out.append(h3)
                elif i == 3: out.append(h4)
                elif i == 4: out.append(h5)
            return out

class PatchMatch(nn.Module):
    """
    PatchMatch algorithm for finding dense correspondences
    Implements randomized nearest neighbor field optimization
    """
    def __init__(self, patch_size=3, iterations=5, alpha=0.5):
        super(PatchMatch, self).__init__()
        self.patch_size = patch_size
        self.iterations = iterations
        self.alpha = alpha
        self.padding = patch_size // 2
        
    def init_nnf(self, source, target):
        """Initialize Nearest Neighbor Field randomly"""
        b, c, h, w = source.shape
        nnf = torch.rand(b, 2, h, w).to(source.device)
        nnf[:, 0] *= (target.shape[2] - 1)  # H coordinates
        nnf[:, 1] *= (target.shape[3] - 1)  # W coordinates
        return nnf
    
    def patch_distance(self, source_patches, target_patches):
        """Compute distance between patches"""
        return torch.sum((source_patches - target_patches) ** 2, dim=[2,3,4])
    
    def propagate(self, nnf, source, target, direction=1):
        """Propagate good matches to neighbors"""
        b, c, h, w = source.shape
        device = source.device
        
        # Create shifted NNF
        shifted_nnf = torch.roll(nnf, shifts=direction, dims=3)
        
        # Get patches for current and shifted positions
        current_patches = self.get_patches(source, nnf)
        shifted_patches = self.get_patches(source, shifted_nnf)
        
        # Compare distances
        current_dist = self.patch_distance(
            current_patches,
            self.get_patches(target, nnf)
        )
        shifted_dist = self.patch_distance(
            shifted_patches,
            self.get_patches(target, shifted_nnf)
        )
        
        # Update NNF where shifted patches give better matches
        mask = (shifted_dist < current_dist).unsqueeze(1)
        nnf = torch.where(mask, shifted_nnf, nnf)
        
        return nnf
    
    def random_search(self, nnf, source, target, radius=4):
        """Randomly search for better matches"""
        b, c, h, w = source.shape
        device = source.device
        
        current_dist = self.patch_distance(
            self.get_patches(source, nnf),
            self.get_patches(target, nnf)
        )
        
        for i in range(radius):
            # Generate random offsets
            rand_offset = torch.randn_like(nnf) * (2 ** -i)
            rand_nnf = nnf + rand_offset
            
            # Clip to valid range
            rand_nnf = torch.clamp(rand_nnf, 0, target.shape[2]-1)
            
            # Compare distances
            rand_dist = self.patch_distance(
                self.get_patches(source, rand_nnf),
                self.get_patches(target, rand_nnf)
            )
            
            # Update NNF where random search gives better matches
            mask = (rand_dist < current_dist).unsqueeze(1)
            nnf = torch.where(mask, rand_nnf, nnf)
            current_dist = torch.where(mask.squeeze(1), rand_dist, current_dist)
            
        return nnf
    
    def forward(self, source, target):
        """
        Find dense correspondences between source and target features
        
        Args:
            source: Source feature maps
            target: Target feature maps
            
        Returns:
            Tensor: Nearest Neighbor Field
        """
        # Initialize NNF
        nnf = self.init_nnf(source, target)
        
        # Iterative optimization
        for _ in range(self.iterations):
            # Forward propagation
            nnf = self.propagate(nnf, source, target, direction=1)
            
            # Backward propagation
            nnf = self.propagate(nnf, source, target, direction=-1)
            
            # Random search
            nnf = self.random_search(nnf, source, target)
            
        return nnf

class DeepImageAnalogy(nn.Module):
    """
    Deep Image Analogy model
    Implements bidirectional dense correspondence finding
    """
    def __init__(self, device='cuda'):
        super(DeepImageAnalogy, self).__init__()
        self.device = device
        self.feature_extractor = FeatureExtractor().to(device)
        self.patch_match = PatchMatch().to(device)
        
        # Weights for different feature levels
        self.level_weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        
    def compute_bidirectional_similarity(self, source_feat, target_feat, nnf):
        """Compute bidirectional similarity between feature maps"""
        # Forward similarity
        forward_patches = self.patch_match.get_patches(source_feat, nnf)
        forward_sim = -self.patch_match.patch_distance(
            forward_patches,
            self.patch_match.get_patches(target_feat, nnf)
        )
        
        # Backward similarity
        backward_nnf = self.patch_match(target_feat, source_feat)
        backward_patches = self.patch_match.get_patches(target_feat, backward_nnf)
        backward_sim = -self.patch_match.patch_distance(
            backward_patches,
            self.patch_match.get_patches(source_feat, backward_nnf)
        )
        
        return (forward_sim + backward_sim) * 0.5
    
    def reconstruct_features(self, source_feat, target_feat, nnf):
        """Reconstruct features using nearest neighbor field"""
        b, c, h, w = source_feat.shape
        device = source_feat.device
        
        # Get matched patches
        matched_patches = self.patch_match.get_patches(target_feat, nnf)
        
        # Reconstruct features
        reconstructed = torch.zeros_like(source_feat)
        count = torch.zeros_like(source_feat)
        
        for i in range(h):
            for j in range(w):
                patch = matched_patches[:, :, i, j]
                y, x = nnf[0, 0, i, j].long(), nnf[0, 1, i, j].long()
                
                # Add patch contribution
                reconstructed[:, :, 
                            max(0, i-self.patch_match.padding):min(h, i+self.patch_match.padding+1),
                            max(0, j-self.patch_match.padding):min(w, j+self.patch_match.padding+1)] += patch
                
                # Update count
                count[:, :,
                      max(0, i-self.patch_match.padding):min(h, i+self.patch_match.padding+1),
                      max(0, j-self.patch_match.padding):min(w, j+self.patch_match.padding+1)] += 1
                
        # Average overlapping regions
        reconstructed = reconstructed / (count + 1e-8)
        return reconstructed
    
    def forward(self, source, target):
        """
        Create deep image analogy between source and target images
        
        Args:
            source: Source image tensor
            target: Target image tensor
            
        Returns:
            tuple: (Reconstructed source, Reconstructed target)
        """
        # Extract features at multiple levels
        source_features = self.feature_extractor(source)
        target_features = self.feature_extractor(target)
        
        # Process features from coarse to fine
        nnf = None
        reconstructed_source = source
        reconstructed_target = target
        
        for level in range(len(source_features)-1, -1, -1):
            # Get current level features
            source_feat = source_features[level]
            target_feat = target_features[level]
            
            # Upsample NNF if available
            if nnf is not None:
                nnf = F.interpolate(nnf, size=source_feat.shape[2:], mode='bilinear')
            
            # Compute dense correspondence
            nnf = self.patch_match(source_feat, target_feat, initial_nnf=nnf)
            
            # Compute bidirectional similarity
            similarity = self.compute_bidirectional_similarity(
                source_feat, target_feat, nnf)
            
            # Reconstruct features
            reconstructed_source = self.reconstruct_features(
                source_feat, target_feat, nnf)
            reconstructed_target = self.reconstruct_features(
                target_feat, source_feat, nnf.flip(1))
            
            # Weight contribution by level
            weight = self.level_weights[level]
            reconstructed_source = (1 - weight) * source_feat + weight * reconstructed_source
            reconstructed_target = (1 - weight) * target_feat + weight * reconstructed_target
        
        return reconstructed_source, reconstructed_target

class DeepImageAnalogyTrainer:
    """
    Trainer class for Deep Image Analogy
    Handles training and inference
    """
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.model = DeepImageAnalogy(self.device).to(self.device)
        
    def create_analogy(self, source_image, target_image):
        """
        Create analogy between source and target images
        
        Args:
            source_image: Source image tensor
            target_image: Target image tensor
            
        Returns:
            tuple: (Analogous source image, Analogous target image)
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(source_image, target_image)
    
    def compute_loss(self, reconstructed, target):
        """Compute reconstruction loss"""
        return F.mse_loss(reconstructed, target)
    
    def train_step(self, source_images, target_images):
        """
        Perform one training step
        
        Args:
            source_images: Batch of source images
            target_images: Batch of target images
            
        Returns:
            dict: Dictionary containing loss values
        """
        self.model.train()
        
        # Create analogies
        reconstructed_source, reconstructed_target = self.model(
            source_images, target_images)
        
        # Compute losses
        source_loss = self.compute_loss(reconstructed_source, source_images)
        target_loss = self.compute_loss(reconstructed_target, target_images)
        total_loss = source_loss + target_loss
        
        # Optimization step
        total_loss.backward()
        
        return {
            'source_loss': source_loss.item(),
            'target_loss': target_loss.item(),
            'total_loss': total_loss.item()
        } 