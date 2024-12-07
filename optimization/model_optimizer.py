import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic
import copy

class ModelOptimizer:
    def __init__(self, model, config=None):
        self.original_model = model
        self.model = copy.deepcopy(model)
        self.config = config or {}
        self.optimization_history = []
        
    def reset_model(self):
        """Reset model to original state"""
        self.model = copy.deepcopy(self.original_model)
        
    def apply_structured_pruning(self, amount=0.3):
        """Apply structured pruning to reduce model size"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=amount,
                    n=2,
                    dim=0  # Prune output channels
                )
                prune.remove(module, 'weight')
                
        self.optimization_history.append({
            'type': 'structured_pruning',
            'amount': amount
        })
        
    def apply_unstructured_pruning(self, amount=0.3):
        """Apply unstructured pruning"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(
                    module,
                    name='weight',
                    amount=amount
                )
                prune.remove(module, 'weight')
                
        self.optimization_history.append({
            'type': 'unstructured_pruning',
            'amount': amount
        })
    
    def quantize_dynamic(self):
        """Apply dynamic quantization"""
        self.model = quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        self.optimization_history.append({
            'type': 'dynamic_quantization'
        })
    
    def optimize_for_inference(self, optimization_level='medium'):
        """Apply various optimization techniques based on level"""
        self.model.eval()
        
        optimization_configs = {
            'light': {
                'pruning_amount': 0.1,
                'quantize': False
            },
            'medium': {
                'pruning_amount': 0.3,
                'quantize': True
            },
            'aggressive': {
                'pruning_amount': 0.5,
                'quantize': True
            }
        }
        
        config = optimization_configs[optimization_level]
        
        # Apply optimizations
        if config['pruning_amount'] > 0:
            self.apply_structured_pruning(config['pruning_amount'])
            
        if config['quantize']:
            self.quantize_dynamic()
            
        return self.model
    
    def get_model_size(self):
        """Get model size in MB"""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        size_all_mb = (param_size + buffer_size) / 1024 / 1024
        return size_all_mb
    
    def get_optimization_summary(self):
        """Get summary of applied optimizations"""
        original_size = self.get_model_size()
        optimized_size = self.get_model_size()
        
        return {
            'original_size_mb': original_size,
            'optimized_size_mb': optimized_size,
            'size_reduction_percent': (original_size - optimized_size) / original_size * 100,
            'optimization_history': self.optimization_history
        }