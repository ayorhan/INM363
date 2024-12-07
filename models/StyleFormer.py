"""
StyleFormer: Transformer-based Style Transfer

Source: Inspired by Vision Transformer architecture
Type: Transformer-based Neural Network

Architecture:
- Vision Transformer encoder
- Style token learning
- Cross-attention style mixing
- Multi-head self-attention
- Feed-forward style transfer

Pros:
- Global style understanding
- Better long-range dependencies
- Flexible style manipulation
- Good feature mixing

Cons:
- High computational complexity
- Large model size
- Requires significant training data
- Slower inference than CNNs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class StyleFormer(nn.Module):
    """
    Transformer-based style transfer model
    """
    def __init__(self, config):
        super(StyleFormer, self).__init__()
        self.device = config['device']
        
        # Initialize components
        self.encoder = StyleEncoder(
            img_size=config['img_size'],
            patch_size=config['patch_size'],
            embed_dim=config['embed_dim']
        ).to(self.device)
        
        self.transformer = StyleTransformer(
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            num_heads=config['num_heads']
        ).to(self.device)
        
        self.decoder = StyleDecoder(
            img_size=config['img_size'],
            patch_size=config['patch_size'],
            embed_dim=config['embed_dim']
        ).to(self.device)
        
        # Style token
        self.style_token = nn.Parameter(
            torch.randn(1, 1, config['embed_dim']))
        
    def forward(self, content_image, style_image):
        # Encode images
        content_tokens = self.encoder(content_image)
        style_tokens = self.encoder(style_image)
        
        # Expand style token
        batch_size = content_image.size(0)
        style_token = self.style_token.expand(batch_size, -1, -1)
        
        # Concatenate tokens
        tokens = torch.cat([content_tokens, style_tokens, style_token], dim=1)
        
        # Apply transformer
        transformed = self.transformer(tokens)
        
        # Decode
        output = self.decoder(transformed[:, :content_tokens.size(1)])
        return output

class StyleEncoder(nn.Module):
    """
    Transformer encoder for style transfer
    """
    def __init__(self, img_size=256, patch_size=16, embed_dim=512):
        super(StyleEncoder, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim))
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Extract patches
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Normalize
        x = self.norm(x)
        return x

class StyleTransformer(nn.Module):
    """
    Main transformer for style transfer
    """
    def __init__(self, embed_dim=512, depth=6, num_heads=8):
        super(StyleTransformer, self).__init__()
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads
            )
            for _ in range(depth)
        ])
        
        # Style attention layers
        self.style_attention = nn.ModuleList([
            StyleAttention(embed_dim)
            for _ in range(depth)
        ])
        
    def forward(self, x):
        # Apply transformer layers with style attention
        for layer, style_attn in zip(self.layers, self.style_attention):
            x = layer(x)
            x = style_attn(x)
        return x

class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and FFN
    """
    def __init__(self, dim, num_heads):
        super(TransformerBlock, self).__init__()
        
        # Multi-head attention
        self.attn = MultiHeadAttention(dim, num_heads)
        
        # Feed-forward network
        self.ffn = FeedForward(dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.norm1(x))
        
        # Feed-forward
        x = x + self.ffn(self.norm2(x))
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism
    """
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Linear projections
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class StyleAttention(nn.Module):
    """
    Style-specific attention mechanism
    """
    def __init__(self, dim):
        super(StyleAttention, self).__init__()
        
        self.style_query = nn.Linear(dim, dim)
        self.style_key = nn.Linear(dim, dim)
        self.style_value = nn.Linear(dim, dim)
        
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        # Split content and style tokens
        content_tokens = x[:, :-1]
        style_token = x[:, -1:]
        
        # Compute style attention
        q = self.style_query(style_token)
        k = self.style_key(content_tokens)
        v = self.style_value(content_tokens)
        
        # Attention weights
        attn = (q @ k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        attn = attn.softmax(dim=-1)
        
        # Apply attention
        style_features = attn @ v
        style_features = self.proj(style_features)
        
        # Combine with content
        x = torch.cat([content_tokens + style_features, style_token], dim=1)
        return x

class StyleDecoder(nn.Module):
    """
    Transformer decoder for image reconstruction
    """
    def __init__(self, img_size=256, patch_size=16, embed_dim=512):
        super(StyleDecoder, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Upsampling layers
        self.layers = nn.ModuleList([
            UpBlock(embed_dim, embed_dim // 2),
            UpBlock(embed_dim // 2, embed_dim // 4),
            UpBlock(embed_dim // 4, embed_dim // 8)
        ])
        
        # Final convolution
        self.final = nn.Sequential(
            nn.Conv2d(embed_dim // 8, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Reshape tokens to 2D
        B = x.size(0)
        h = w = int(np.sqrt(x.size(1)))
        x = x.transpose(1, 2).view(B, -1, h, w)
        
        # Progressive upsampling
        for layer in self.layers:
            x = layer(x)
            
        # Final convolution
        x = self.final(x)
        return x

class UpBlock(nn.Module):
    """
    Upsampling block for decoder
    """
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class FeedForward(nn.Module):
    """
    Simple feed-forward network with GELU activation
    """
    def __init__(self, dim):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        return self.net(x)

class VGGPerceptual(nn.Module):
    """VGG-based perceptual loss network"""
    def __init__(self):
        super(VGGPerceptual, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.content_layer = vgg[:20]
        self.style_layers = [vgg[:3], vgg[:8], vgg[:13], vgg[:22], vgg[:31]]
        
        # Freeze parameters
        for params in self.parameters():
            params.requires_grad = False
            
    def forward(self, x):
        content = self.content_layer(x)
        style = [layer(x) for layer in self.style_layers]
        return {'content': content, 'style': style}

class StyleFormerModel:
    """
    Complete StyleFormer model with training and inference
    """
    def __init__(self, config):
        self.device = config['device']
        self.config = config
        
        # Initialize StyleFormer
        self.model = StyleFormer(config).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['lr']
        )
        
        # VGG for perceptual loss
        self.vgg = VGGPerceptual().to(self.device)
        
    def train_step(self, content_images, style_images):
        """
        Perform one training step
        """
        self.optimizer.zero_grad()
        
        # Generate stylized image
        output = self.model(content_images, style_images)
        
        # Compute losses
        content_loss = self.compute_content_loss(
            output, content_images)
        style_loss = self.compute_style_loss(
            output, style_images)
        
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
    
    def compute_content_loss(self, output, content_images):
        """Compute content loss using VGG features"""
        content_features = self.vgg(content_images)
        output_features = self.vgg(output)
        
        return F.mse_loss(
            output_features['content'],
            content_features['content']
        )
    
    def compute_style_loss(self, output, style_images):
        """Compute style loss using gram matrices"""
        style_features = self.vgg(style_images)
        output_features = self.vgg(output)
        
        style_loss = 0
        for sf, of in zip(style_features['style'], output_features['style']):
            style_loss += F.mse_loss(
                self.gram_matrix(of),
                self.gram_matrix(sf)
            )
            
        return style_loss
    
    def gram_matrix(self, x):
        """Compute gram matrix"""
        b, c, h, w = x.size()
        features = x.view(b, c, -1)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def style_transfer(self, content_image, style_image):
        """
        Perform style transfer
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(content_image, style_image) 