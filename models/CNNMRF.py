"""
CNNMRF: Combining Markov Random Fields and CNNs for Image Synthesis

Source: https://arxiv.org/abs/1601.04589
Type: CNN (Convolutional Neural Network)

Architecture:
- VGG-based feature extraction
- MRF-based patch matching
- Multi-scale synthesis
- Patch-based optimization
- Local pattern matching

Pros:
- Better local pattern preservation
- Detailed texture synthesis
- Flexible style control
- Good structure preservation

Cons:
- Slow optimization process
- High memory consumption
- Complex patch matching
- Parameter sensitivity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class VGGFeatures(nn.Module):
    """
    VGG feature extractor for CNNMRF
    Extracts multi-level features for patch matching
    """
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg = models.vgg19(pretrained=True).features.eval()
        
        # Create feature extraction layers
        self.slice1 = nn.Sequential(*list(vgg)[:4])   # relu1_1
        self.slice2 = nn.Sequential(*list(vgg)[4:9])  # relu2_1
        self.slice3 = nn.Sequential(*list(vgg)[9:18]) # relu3_1
        self.slice4 = nn.Sequential(*list(vgg)[18:27])# relu4_1
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x, layers=None):
        """
        Extract features at specified layers
        
        Args:
            x: Input image tensor
            layers: List of layer indices to extract features from
            
        Returns:
            list: List of feature maps
        """
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        
        if layers is None:
            return [h1, h2, h3, h4]
        
        outputs = []
        for layer in layers:
            if layer == 0: outputs.append(h1)
            elif layer == 1: outputs.append(h2)
            elif layer == 2: outputs.append(h3)
            elif layer == 3: outputs.append(h4)
        return outputs

class PatchMatcher(nn.Module):
    """
    Patch matching module for CNNMRF
    Implements efficient patch matching using GPU
    """
    def __init__(self, patch_size=3, stride=1):
        super(PatchMatcher, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        
    def extract_patches(self, features):
        """Extract patches from feature maps"""
        b, c, h, w = features.size()
        patches = F.unfold(features, 
                          kernel_size=self.patch_size,
                          stride=self.stride,
                          padding=self.patch_size//2)
        patches = patches.view(b, c, self.patch_size, self.patch_size, -1)
        patches = patches.permute(0, 4, 1, 2, 3)  # [B, N, C, H, W]
        return patches
    
    def compute_distances(self, content_patches, style_patches):
        """Compute normalized distances between patch sets"""
        # Reshape patches for efficient computation
        c_patches = content_patches.view(content_patches.size(0), -1, 
                                       content_patches.size(2) * 
                                       content_patches.size(3) * 
                                       content_patches.size(4))
        s_patches = style_patches.view(style_patches.size(0), -1,
                                     style_patches.size(2) * 
                                     style_patches.size(3) * 
                                     style_patches.size(4))
        
        # Normalize patches
        c_patches = F.normalize(c_patches, dim=2)
        s_patches = F.normalize(s_patches, dim=2)
        
        # Compute distances
        distances = torch.bmm(c_patches, s_patches.transpose(1, 2))
        return distances
    
    def find_best_matches(self, distances):
        """Find best matching style patches for each content patch"""
        best_matches = distances.max(dim=2)[1]
        return best_matches
    
    def reconstruct_features(self, style_patches, best_matches, output_size):
        """Reconstruct features using matched patches"""
        b, n, c, h, w = style_patches.size()
        matched_patches = style_patches[torch.arange(b).unsqueeze(1), best_matches]
        
        # Reshape for unfolding
        matched_patches = matched_patches.view(b, -1, c * h * w)
        
        # Reconstruct features
        output = F.fold(matched_patches,
                       output_size=output_size,
                       kernel_size=self.patch_size,
                       stride=self.stride,
                       padding=self.patch_size//2)
        
        # Normalize by overlap count
        ones = torch.ones_like(matched_patches)
        divisor = F.fold(ones,
                        output_size=output_size,
                        kernel_size=self.patch_size,
                        stride=self.stride,
                        padding=self.patch_size//2)
        
        return output / (divisor + 1e-8)

class CNNMRF(nn.Module):
    """
    CNN-based Markov Random Field model for style transfer
    """
    def __init__(self, device='cuda'):
        super(CNNMRF, self).__init__()
        self.device = device
        self.vgg = VGGFeatures().to(device)
        self.patch_matcher = PatchMatcher()
        
        # Initialize patch sizes for different layers
        self.patch_sizes = {
            0: 3,  # relu1_1
            1: 3,  # relu2_1
            2: 3,  # relu3_1
            3: 3   # relu4_1
        }
        
    def mrf_loss(self, content_feats, style_feats, layer):
        """Compute MRF loss for a specific layer"""
        patch_size = self.patch_sizes[layer]
        self.patch_matcher.patch_size = patch_size
        
        # Extract patches
        content_patches = self.patch_matcher.extract_patches(content_feats)
        style_patches = self.patch_matcher.extract_patches(style_feats)
        
        # Compute patch distances and find best matches
        distances = self.patch_matcher.compute_distances(content_patches, style_patches)
        best_matches = self.patch_matcher.find_best_matches(distances)
        
        # Reconstruct features
        reconstructed = self.patch_matcher.reconstruct_features(
            style_patches, best_matches, content_feats.size()[2:])
        
        return F.mse_loss(content_feats, reconstructed)
    
    def content_loss(self, content_feats, target_feats):
        """Compute content loss"""
        return F.mse_loss(content_feats, target_feats)
    
    def total_variation_loss(self, image):
        """Compute total variation loss for smoothness"""
        loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
               torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
        return loss

class CNNMRFTrainer:
    """
    Trainer class for CNNMRF
    Handles optimization and style transfer
    """
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.model = CNNMRF(self.device)
        
    def transfer_style(self, content_image, style_image, num_steps=500):
        """
        Perform style transfer
        
        Args:
            content_image: Content image tensor
            style_image: Style image tensor
            num_steps: Number of optimization steps
            
        Returns:
            Tensor: Stylized image
        """
        # Initialize with content image
        target = content_image.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([target], lr=1.0)
        
        # Extract style features
        style_features = self.model.vgg(style_image)
        content_features = self.model.vgg(content_image)
        
        for step in range(num_steps):
            def closure():
                optimizer.zero_grad()
                
                # Extract features from current result
                target_features = self.model.vgg(target)
                
                # Compute MRF losses for each layer
                mrf_loss = 0
                for i in range(len(target_features)):
                    mrf_loss += self.config['mrf_weights'][i] * self.model.mrf_loss(
                        target_features[i], style_features[i], i)
                
                # Compute content loss
                content_loss = self.config['content_weight'] * self.model.content_loss(
                    target_features[self.config['content_layer']],
                    content_features[self.config['content_layer']])
                
                # Compute total variation loss
                tv_loss = self.config['tv_weight'] * self.model.total_variation_loss(target)
                
                # Total loss
                total_loss = mrf_loss + content_loss + tv_loss
                
                # Backward pass
                total_loss.backward()
                
                # Print progress
                if step % 50 == 0:
                    print(f'Step {step}, Loss: {total_loss.item():.4f}')
                
                return total_loss
            
            optimizer.step(closure)
        
        return target.detach()
    
    def style_transfer_with_guidance(self, content_image, style_image, 
                                   content_mask=None, style_mask=None,
                                   num_steps=500):
        """
        Perform guided style transfer with semantic masks
        
        Args:
            content_image: Content image tensor
            style_image: Style image tensor
            content_mask: Semantic mask for content image
            style_mask: Semantic mask for style image
            num_steps: Number of optimization steps
            
        Returns:
            Tensor: Stylized image
        """
        # Initialize with content image
        target = content_image.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([target], lr=1.0)
        
        # Extract features
        style_features = self.model.vgg(style_image)
        content_features = self.model.vgg(content_image)
        
        for step in range(num_steps):
            def closure():
                optimizer.zero_grad()
                target_features = self.model.vgg(target)
                
                # Compute masked MRF losses
                mrf_loss = 0
                for i in range(len(target_features)):
                    if content_mask is not None and style_mask is not None:
                        # Scale masks to feature size
                        c_mask = F.interpolate(content_mask, target_features[i].shape[2:])
                        s_mask = F.interpolate(style_mask, style_features[i].shape[2:])
                        
                        # Apply masks to features
                        target_masked = target_features[i] * c_mask
                        style_masked = style_features[i] * s_mask
                        
                        mrf_loss += self.config['mrf_weights'][i] * self.model.mrf_loss(
                            target_masked, style_masked, i)
                    else:
                        mrf_loss += self.config['mrf_weights'][i] * self.model.mrf_loss(
                            target_features[i], style_features[i], i)
                
                # Other losses remain the same
                content_loss = self.config['content_weight'] * self.model.content_loss(
                    target_features[self.config['content_layer']],
                    content_features[self.config['content_layer']])
                
                tv_loss = self.config['tv_weight'] * self.model.total_variation_loss(target)
                
                total_loss = mrf_loss + content_loss + tv_loss
                total_loss.backward()
                
                return total_loss
            
            optimizer.step(closure)
        
        return target.detach() 