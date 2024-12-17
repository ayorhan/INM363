"""
Configuration management for style transfer models
"""

import yaml
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    batch_size: int = 4
    num_epochs: int = 100
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    num_workers: int = 4
    save_interval: int = 1000
    validation_interval: int = 500
    min_lr: float = 0.00001
    
@dataclass
class ModelConfig:
    """Model architecture configuration"""
    model_type: str  # e.g., 'adain', 'cyclegan', 'pix2pix'
    input_channels: int = 3
    output_channels: int = 3
    base_filters: int = 64
    use_dropout: bool = True
    
@dataclass
class DataConfig:
    """Dataset configuration"""
    dataset_path: str = "data/"
    content_path: str = "data/coco"
    style_path: str = "data/style"
    image_size: int = 256
    crop_size: int = 224
    train_content_size: int = 20000
    train_style_size: int = 1500
    val_content_size: int = 2000
    val_style_size: int = 150
    use_augmentation: bool = True
    num_workers: int = 4

@dataclass
class LoggingConfig:
    """Logging configuration parameters"""
    use_wandb: bool = False
    project_name: str = "style-transfer"
    run_name: str = "default"
    log_interval: int = 100
    save_dir: str = "checkpoints"
    output_dir: str = "outputs"
    
class StyleTransferConfig:
    """Complete configuration for style transfer training"""
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.training = TrainingConfig()
        self.model = ModelConfig(model_type="adain")
        self.data = DataConfig()
        self.logging = LoggingConfig()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Update all config sections
        if 'logging' in config_dict:
            self.logging = LoggingConfig(**config_dict['logging'])
            
        # Update other sections
        for section in ['training', 'model', 'data']:
            if section in config_dict:
                config_obj = getattr(self, section)
                for key, value in config_dict[section].items():
                    setattr(config_obj, key, value)
    
    def save_config(self, save_path: str) -> None:
        """Save configuration to YAML file"""
        config_dict = {
            'training': self.training.__dict__,
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'logging': self.logging.__dict__
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return self.model.__dict__
    
    def get_optimizer_config(self) -> Dict[str, Any]:
        """Get optimizer configuration"""
        return {
            'lr': self.training.learning_rate,
            'betas': (self.training.beta1, self.training.beta2)
        }
    
    def get_dataloader_config(self) -> Dict[str, Any]:
        """Get dataloader configuration"""
        return {
            'batch_size': self.training.batch_size,
            'num_workers': self.training.num_workers,
            'image_size': self.data.image_size,
            'crop_size': self.data.crop_size,
            'use_augmentation': self.data.use_augmentation
        }

def load_config(config_path: str) -> StyleTransferConfig:
    """Helper function to load configuration"""
    return StyleTransferConfig(config_path)