"""
StyleGAN-NADA: Zero-Shot Domain Adaptation using CLIP

Source: https://arxiv.org/abs/2108.00946
Type: GAN (Generative Adversarial Network)

Architecture:
- StyleGAN2 backbone
- CLIP-based text guidance
- Directional CLIP loss
- Identity preservation
- Multi-scale discriminator

Pros:
- Text-driven domain adaptation
- No training pairs needed
- High quality results
- Flexible control

Cons:
- Computationally intensive
- Large model size
- Training instability
- CLIP limitations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import clip
from collections import OrderedDict

class StyleGANGenerator(nn.Module):
    """
    Modified StyleGAN2 generator with NADA adaptations
    """
    def __init__(self, z_dim=512, w_dim=512, c_dim=0, img_resolution=1024, img_channels=3):
        super(StyleGANGenerator, self).__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        
        # Mapping network
        self.mapping = MappingNetwork(
            z_dim=z_dim,
            w_dim=w_dim,
            c_dim=c_dim,
            num_layers=8
        )
        
        # Synthesis network
        self.synthesis = SynthesisNetwork(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels
        )
        
    def forward(self, z, c=None, truncation_psi=1.0, truncation_cutoff=None):
        w = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(w)
        return img

class MappingNetwork(nn.Module):
    """
    Mapping network to transform latent vectors
    """
    def __init__(self, z_dim, w_dim, c_dim, num_layers=8):
        super(MappingNetwork, self).__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.c_dim = c_dim
        self.num_layers = num_layers
        
        # Layers
        features = [z_dim + c_dim] + [w_dim] * num_layers
        
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(features[:-1], features[1:]):
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.LeakyReLU(0.2))
            
        # Register buffer for truncation
        self.register_buffer('w_avg', torch.zeros([w_dim]))
        
    def forward(self, z, c=None, truncation_psi=1.0, truncation_cutoff=None):
        # Embed conditioning if provided
        if self.c_dim > 0:
            z = torch.cat([z, c], dim=1)
            
        # Execute layers
        x = z
        for layer in self.layers:
            x = layer(x)
            
        # Apply truncation
        if truncation_psi != 1.0:
            w = self.w_avg.lerp(x, truncation_psi)
            if truncation_cutoff is None:
                return w
            return torch.where(x.abs() <= truncation_cutoff, w, x)
            
        return x

class NADA(nn.Module):
    """
    NADA model for text-driven style transfer
    """
    def __init__(self, config):
        super(NADA, self).__init__()
        self.config = config
        
        # Load pretrained StyleGAN
        self.generator = StyleGANGenerator()
        
        # Load CLIP model
        self.clip_model, _ = clip.load(config['clip_model'], device=config['device'])
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Initialize domain adaptation layers
        self.domain_adapter = DomainAdapter(
            w_dim=self.generator.w_dim,
            clip_dim=self.clip_model.visual.output_dim
        )
        
    def get_text_features(self, text_prompts):
        """Extract CLIP text features"""
        text_tokens = clip.tokenize(text_prompts).to(self.config['device'])
        text_features = self.clip_model.encode_text(text_tokens)
        return F.normalize(text_features, dim=-1)
    
    def get_image_features(self, images):
        """Extract CLIP image features"""
        image_features = self.clip_model.encode_image(images)
        return F.normalize(image_features, dim=-1)
    
    def forward(self, z, text_prompt, truncation_psi=0.7):
        """
        Generate images conditioned on text prompt
        
        Args:
            z: Latent vectors
            text_prompt: Text description of target style
            truncation_psi: Truncation parameter
            
        Returns:
            Tensor: Generated images
        """
        # Get text features
        text_features = self.get_text_features([text_prompt])
        
        # Generate initial w vectors
        w = self.generator.mapping(z, truncation_psi=truncation_psi)
        
        # Apply domain adaptation
        w_adapted = self.domain_adapter(w, text_features)
        
        # Generate images
        images = self.generator.synthesis(w_adapted)
        return images

class DomainAdapter(nn.Module):
    """
    Domain adaptation network for NADA
    """
    def __init__(self, w_dim, clip_dim):
        super(DomainAdapter, self).__init__()
        
        self.mapper = nn.Sequential(
            nn.Linear(w_dim + clip_dim, w_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(w_dim * 2, w_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(w_dim, w_dim)
        )
        
    def forward(self, w, text_features):
        # Concatenate w vectors with text features
        x = torch.cat([w, text_features.repeat(w.size(0), 1)], dim=1)
        
        # Generate adaptation parameters
        delta_w = self.mapper(x)
        
        # Apply adaptation
        w_adapted = w + delta_w
        return w_adapted

class NADATrainer:
    """
    Trainer class for StyleGAN-NADA
    """
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        
        # Initialize model
        self.model = NADA(config).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.domain_adapter.parameters(),
            lr=config['learning_rate'],
            betas=(config['beta1'], 0.999)
        )
        
        # Initialize loss functions
        self.directional_loss = DirectionalCLIPLoss()
        self.identity_loss = IdentityLoss()
        
    def train_step(self, z, source_prompt, target_prompt):
        """
        Perform one training step
        
        Args:
            z: Latent vectors
            source_prompt: Source domain text description
            target_prompt: Target domain text description
            
        Returns:
            dict: Dictionary containing loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Generate images
        generated_images = self.model(z, target_prompt)
        
        # Get CLIP features
        source_features = self.model.get_text_features([source_prompt])
        target_features = self.model.get_text_features([target_prompt])
        image_features = self.model.get_image_features(generated_images)
        
        # Compute losses
        dir_loss = self.directional_loss(
            image_features, source_features, target_features)
        id_loss = self.identity_loss(generated_images, z)
        
        # Total loss
        total_loss = (self.config['lambda_dir'] * dir_loss +
                     self.config['lambda_id'] * id_loss)
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'dir_loss': dir_loss.item(),
            'id_loss': id_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def generate_images(self, z, text_prompt):
        """
        Generate images using trained model
        
        Args:
            z: Latent vectors
            text_prompt: Text description of target style
            
        Returns:
            Tensor: Generated images
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(z, text_prompt)

class DirectionalCLIPLoss(nn.Module):
    """
    Directional CLIP loss for text-guided adaptation
    """
    def forward(self, image_features, source_features, target_features):
        source_direction = F.normalize(target_features - source_features, dim=-1)
        image_direction = F.normalize(image_features - source_features, dim=-1)
        
        return -torch.cosine_similarity(source_direction, image_direction).mean()

class IdentityLoss(nn.Module):
    """
    Identity preservation loss
    """
    def __init__(self):
        super(IdentityLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features.eval()
        
    def forward(self, generated_images, original_z):
        with torch.no_grad():
            original_images = self.model.generator(original_z)
            
        gen_features = self.vgg(generated_images)
        orig_features = self.vgg(original_images)
        
        return F.mse_loss(gen_features, orig_features)
class SynthesisNetwork(nn.Module):
    """
    StyleGAN2 synthesis network
    """
    def __init__(self, w_dim, img_resolution, img_channels):
        super(SynthesisNetwork, self).__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        
        # Basic implementation - you may want to expand this
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(w_dim, 512, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, img_channels, 3, 1, 1)
        )
        
    def forward(self, w):
        # Reshape w for convolution
        x = w.view(-1, self.w_dim, 1, 1)
        return self.conv(x)
