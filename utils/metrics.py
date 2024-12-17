"""
Evaluation metrics for style transfer models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from scipy import linalg
import lpips
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union

class StyleTransferMetrics:
    """Collection of metrics for style transfer evaluation"""
    def __init__(self, device='cpu'):
        # Check if MPS is available (for Apple Silicon)
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        # Initialize LPIPS
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        
        # Initialize VGG for content/style metrics
        self.vgg = self._load_vgg().to(self.device)
        
        # Initialize FID
        self.inception = self._load_inception().to(self.device)
        
    def _load_vgg(self) -> nn.Module:
        """Load and prepare VGG model"""
        vgg = models.vgg19(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        return vgg
    
    def _load_inception(self) -> nn.Module:
        """Load and prepare Inception model for FID"""
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.fc = nn.Identity()
        inception.eval()
        for param in inception.parameters():
            param.requires_grad = False
        return inception
    
    def compute_content_loss(self, 
                           generated: torch.Tensor, 
                           target: torch.Tensor,
                           layer_idx: int = 22) -> float:
        """Compute content loss using VGG features"""
        # Move inputs to the same device as the model
        generated = generated.to(self.device)
        target = target.to(self.device)
        features_generated = self.vgg[:layer_idx](generated)
        features_target = self.vgg[:layer_idx](target)
        return F.mse_loss(features_generated, features_target).item()
    
    def compute_style_loss(self,
                          generated: torch.Tensor,
                          style: torch.Tensor,
                          layer_indices: List[int] = [3, 8, 13, 22, 31]) -> float:
        """Compute style loss using Gram matrices"""
        # Move inputs to the same device as the model
        generated = generated.to(self.device)
        style = style.to(self.device)
        
        style_loss = 0
        for layer_idx in layer_indices:
            features_generated = self.vgg[:layer_idx](generated)
            features_style = self.vgg[:layer_idx](style)
            
            gram_generated = self._gram_matrix(features_generated)
            gram_style = self._gram_matrix(features_style)
            
            style_loss += F.mse_loss(gram_generated, gram_style)
            
        return style_loss.item() / len(layer_indices)
    
    def _gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix"""
        b, c, h, w = x.size()
        features = x.view(b, c, -1)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def compute_lpips(self, 
                     generated: torch.Tensor, 
                     target: torch.Tensor) -> float:
        """Compute LPIPS perceptual distance"""
        # Move inputs to the same device as the model
        generated = generated.to(self.device)
        target = target.to(self.device)
        
        # Ensure the scaling layer is on the same device
        if hasattr(self.lpips_fn, 'scaling_layer'):
            self.lpips_fn.scaling_layer = self.lpips_fn.scaling_layer.to(self.device)
        
        with torch.no_grad():
            distance = self.lpips_fn(generated, target)
        return distance.mean().item()
    
    def compute_fid(self, 
                   generated_features: torch.Tensor, 
                   target_features: torch.Tensor) -> float:
        """Compute Fréchet Inception Distance"""
        mu1, sigma1 = self._get_statistics(generated_features)
        mu2, sigma2 = self._get_statistics(target_features)
        
        diff = mu1 - mu2
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        tr_covmean = np.trace(covmean)
        
        return (diff.dot(diff) + 
                np.trace(sigma1) + 
                np.trace(sigma2) - 
                2 * tr_covmean)
    
    def _get_statistics(self, features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate mean and covariance statistics"""
        features = features.cpu().numpy()
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def compute_psnr(self, 
                     generated: torch.Tensor, 
                     target: torch.Tensor,
                     max_value: float = 1.0) -> float:
        """Compute Peak Signal-to-Noise Ratio"""
        mse = F.mse_loss(generated, target).item()
        if mse == 0:
            return float('inf')
        return 20 * np.log10(max_value) - 10 * np.log10(mse)
    
    def compute_ssim(self, 
                    generated: torch.Tensor, 
                    target: torch.Tensor,
                    window_size: int = 11) -> float:
        """Compute Structural Similarity Index"""
        C1 = (0.01 * 1) ** 2
        C2 = (0.03 * 1) ** 2
        
        # Create window
        window = self._create_window(window_size).to(self.device)
        
        # Ensure window is on the same device and type as the input
        window = window.to(generated.device, generated.dtype)
        
        mu1 = F.conv2d(generated, window, padding=window_size//2, groups=3)
        mu2 = F.conv2d(target, window, padding=window_size//2, groups=3)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(generated * generated, window, padding=window_size//2, groups=3) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=3) - mu2_sq
        sigma12 = F.conv2d(generated * target, window, padding=window_size//2, groups=3) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                  ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()
    
    def _create_window(self, window_size: int) -> torch.Tensor:
        """Create a Gaussian window for SSIM"""
        sigma = 1.5
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2 * sigma**2)) 
                            for x in range(window_size)])
        gauss = gauss / gauss.sum()
        
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t())
        _3D_window = _2D_window.unsqueeze(0).repeat(3, 1, 1, 1)
        
        return _3D_window

def compute_metrics(outputs: Union[Dict[str, torch.Tensor], torch.Tensor],
                   targets: Union[Dict[str, torch.Tensor], torch.Tensor],
                   metrics_config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Compute all relevant metrics for the current batch
    
    Args:
        outputs: Model outputs (either tensor or dictionary)
        targets: Target values (either tensor or dictionary)
        metrics_config: Optional configuration for metrics
    
    Returns:
        Dictionary of computed metrics
    """
    # Convert tensors to dictionary format if needed
    if isinstance(outputs, torch.Tensor):
        outputs = {'generated': outputs}
    if isinstance(targets, torch.Tensor):
        targets = {'content': targets}
        
    # Ensure outputs and targets are dictionaries
    if not isinstance(outputs, dict):
        raise TypeError("outputs must be a dictionary, got {}".format(type(outputs)))
    if not isinstance(targets, dict):
        raise TypeError("targets must be a dictionary, got {}".format(type(targets)))
        
    if metrics_config is None:
        metrics_config = {'device': 'cuda'}
        
    metrics = StyleTransferMetrics(device=metrics_config['device'])
    results = {}
    
    # Content preservation metrics
    if 'generated' in outputs and 'content' in targets:
        results['content_loss'] = metrics.compute_content_loss(
            outputs['generated'],
            targets['content']
        )
        results['lpips'] = metrics.compute_lpips(
            outputs['generated'],
            targets['content']
        )
        results['psnr'] = metrics.compute_psnr(
            outputs['generated'],
            targets['content']
        )
        results['ssim'] = metrics.compute_ssim(
            outputs['generated'],
            targets['content']
        )
    
    # Style transfer metrics
    if 'generated' in outputs and 'style' in targets:
        results['style_loss'] = metrics.compute_style_loss(
            outputs['generated'],
            targets['style']
        )
    
    # FID score (if enough samples are available)
    if ('generated_features' in outputs and 
        'target_features' in targets):
        results['fid'] = metrics.compute_fid(
            outputs['generated_features'],
            targets['target_features']
        )
    
    return results

class MetricsLogger:
    """Logger for tracking metrics during training"""
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, batch_metrics):
        for key, value in batch_metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0
                self.counts[key] = 0
            self.metrics[key] += value
            self.counts[key] += 1
    
    def get_average(self):
        return {k: self.metrics[k] / self.counts[k] for k in self.metrics}
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.metrics = {} 