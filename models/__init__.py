"""
Model factory for style transfer models
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from utils.config import StyleTransferConfig, ModelConfig

from .CycleGAN import CycleGAN
from .JohnsonModel import JohnsonModel

class ModelRegistry:
    """Registry for all available models"""
    _models = {
        'cyclegan': CycleGAN,
        'johnson': JohnsonModel
    }
    
    @classmethod
    def register(cls, name: str, model_class: nn.Module):
        """Register a new model"""
        cls._models[name.lower()] = model_class
    
    @classmethod
    def get_model(cls, name: str) -> Optional[nn.Module]:
        """Get model class by name"""
        return cls._models.get(name.lower())
    
    @classmethod
    def list_models(cls) -> list:
        """List all registered models"""
        return list(cls._models.keys())

MODEL_MAPPING = {
    'johnson': JohnsonModel,
    'cyclegan': CycleGAN
}

def get_model(config: StyleTransferConfig) -> nn.Module:
    """Get appropriate model based on config"""
    model_config = config.model
    model_type = model_config.model_type
    
    if model_type == 'cyclegan':
        return CycleGAN({
            'input_channels': model_config.input_channels,
            'output_channels': model_config.output_channels,
            'base_filters': model_config.base_filters,
            'n_residual_blocks': model_config.n_residual_blocks,
            'use_dropout': model_config.use_dropout
        })
    else:  # johnson, adain
        return JohnsonModel({
            'input_channels': model_config.input_channels,
            'output_channels': model_config.output_channels,
            'base_filters': model_config.base_filters,
            'n_residuals': model_config.n_residuals,
            'use_dropout': model_config.use_dropout,
            'norm_type': model_config.norm_type,
            'content_layers': model_config.content_layers,
            'style_layers': model_config.style_layers
        })

def load_pretrained_weights(model: nn.Module, weights_path: str) -> None:
    """
    Load pretrained weights into model
    
    Args:
        model: Model to load weights into
        weights_path: Path to weights file
    """
    try:
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load weights
        model.load_state_dict(state_dict)
        print(f"Successfully loaded pretrained weights from {weights_path}")
        
    except Exception as e:
        print(f"Error loading pretrained weights: {str(e)}")
        raise

def get_model_parameters(model_type: str) -> Dict[str, Any]:
    """
    Get default parameters for a specific model
    
    Args:
        model_type: Name of the model
    
    Returns:
        Dictionary of default parameters
    """
    defaults = {
        'cyclegan': {
            'ngf': 64,
            'ndf': 64,
            'n_residual_blocks': 9,
            'use_dropout': True,
        },
        'johnson': {
            'input_channels': 3,
            'base_filters': 64,
            'n_residuals': 5,
            'use_dropout': True,
        }
    }
    
    return defaults.get(model_type.lower(), {})

class ModelBuilder:
    """Helper class for building models with specific configurations"""
    def __init__(self, base_config: Dict[str, Any]):
        self.config = base_config.copy()
    
    def with_params(self, **kwargs) -> 'ModelBuilder':
        """Add additional parameters to configuration"""
        self.config.update(kwargs)
        return self
    
    def with_pretrained(self, weights_path: str) -> 'ModelBuilder':
        """Add pretrained weights path"""
        self.config['pretrained_path'] = weights_path
        return self
    
    def build(self) -> nn.Module:
        """Build and return the model"""
        return get_model(self.config)

# Example usage:
"""
# Simple usage:
model = get_model({'model_type': 'adain', 'device': 'cuda'})

# Using ModelBuilder:
model = (ModelBuilder({'model_type': 'styleformer', 'device': 'cuda'})
         .with_params(img_size=512, patch_size=32)
         .with_pretrained('weights/styleformer_v1.pth')
         .build())
""" 