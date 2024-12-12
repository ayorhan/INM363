"""
Model factory for style transfer models
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from utils.config import StyleTransferConfig

from .AdaIN import AdaIN
from .AdaINPlusPlus import AdaINPlusPlusModel
from .ArtFlow import ArtFlow
from .AttentionGAN import AttentionGAN
from .CartoonGAN import CartoonGAN
from .CNNMRF import CNNMRF
from .CycleGAN import CycleGAN
from .CycleGANVC import CycleGANVC
from .DeepImageAnalogy import DeepImageAnalogy
from .FastPhotoStyle import FastPhotoStyle
from .FUNIT import FUNIT
from .GatysModel import GatysModel
from .JohnsonModel import JohnsonModel
from .LinearStyleTransfer import LinearStyleTransfer
from .LST import LST
from .MSGNet import MSGNet
from .MUNIT import MUNIT
from .NeuralDoodle import NeuralDoodle
from .PhotoWCT import PhotoWCT
from .Pix2Pix import Pix2Pix
from .ReStyle import ReStyle
from .SANet import SANet
from .StarGAN import StarGAN
from .STROTSS import STROTSS
from .StyleFormer import StyleFormer
from .StyleGAN2 import StyleGAN2
#from .StyleGAN2ADA import StyleGAN2ADA
from .StyleGANNADAv2 import StyleGANNADAv2
from .StyleMixer import StyleMixerModel
from .UGATIT import UGATIT


class ModelRegistry:
    """Registry for all available models"""
    _models = {
        'adain': AdaIN,
        'adainplusplus': AdaINPlusPlusModel,
        'artflow': ArtFlow,
        'attentiongan': AttentionGAN,
        'cartoongan': CartoonGAN,
        'cnnmrf': CNNMRF,
        'cyclegan': CycleGAN,
        'cycleganvc': CycleGANVC,
        'deepimageanalogy': DeepImageAnalogy,
        'fastphotostyle': FastPhotoStyle,
        'funit': FUNIT,
        'gatysmodel': GatysModel,
        'johnsonmodel': JohnsonModel,
        'linearstyletransfer': LinearStyleTransfer,
        'lst': LST,
        'msgnet': MSGNet,
        'munit': MUNIT,
        'neuraldoodle': NeuralDoodle,
        'photowct': PhotoWCT,
        'pix2pix': Pix2Pix,
        'restyle': ReStyle,
        'sanet': SANet,
        'stargan': StarGAN,
        'strotss': STROTSS,
        'styleformer': StyleFormer,
        'stylegan2': StyleGAN2,
#        'stylegan2ada': StyleGAN2ADA,
        'stylegannadav2': StyleGANNADAv2,
        'style_mixer': StyleMixerModel,
        'ugatit': UGATIT,
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

def get_model(config: StyleTransferConfig):
    """Get appropriate model based on configuration"""
    model_type = config.model.model_type.lower()
    model_config = config.get_model_config()
    
    MODEL_MAPPING = {
        'adain': lambda: AdaIN(),
        'adainplusplus': lambda: AdaINPlusPlusModel(model_config),
        'stylegan2': lambda: StyleGAN2(model_config),
    #    'stylegan2ada': lambda: StyleGAN2ADA(model_config),
        'style_mixer': lambda: StyleMixerModel(model_config)
    }
    
    if model_type not in MODEL_MAPPING:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    return MODEL_MAPPING[model_type]()

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
        'adain': {
            'encoder_type': 'vgg19',
            'decoder_type': 'basic',
            'content_layers': ['relu4_1'],
            'style_layers': ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'],
        },
        'styleformer': {
            'img_size': 256,
            'patch_size': 16,
            'embed_dim': 512,
            'depth': 6,
            'num_heads': 8,
        },
        'strotss': {
            'content_weight': 1.0,
            'style_weight': 1.0,
            'reg_weight': 1e-4,
        },
        'restyle': {
            'num_iterations': 5,
            'use_attention': True,
            'feature_layers': ['relu1_1', 'relu2_1', 'relu3_1'],
        },
        'cyclegan': {
            'ngf': 64,
            'ndf': 64,
            'n_residual_blocks': 9,
            'use_dropout': True,
        },
        'pix2pix': {
            'input_channels': 3,
            'output_channels': 3,
            'base_filters': 64,
            'n_layers': 3,
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