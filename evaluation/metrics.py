import torch
import torch.nn as nn
import torchvision.models as models
from scipy import linalg
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time
import psutil
from torchvision import transforms

class StyleTransferEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Initialize inception model for FID calculation
        self.inception = models.inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()
        # Remove final classification layer
        self.inception.fc = nn.Identity()
        
        # Standard image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
    def _get_inception_features(self, images):
        """Extract features using InceptionV3"""
        features = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                batch = self.preprocess(batch).to(self.device)
                feat = self.inception(batch)
                features.append(feat.cpu().numpy())
                
        return np.concatenate(features, axis=0)
    
    def calculate_fid(self, real_images, generated_images):
        """Calculate Fréchet Inception Distance"""
        real_features = self._get_inception_features(real_images)
        gen_features = self._get_inception_features(generated_images)
        
        # Calculate mean and covariance statistics
        mu1, sigma1 = real_features.mean(0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = gen_features.mean(0), np.cov(gen_features, rowvar=False)
        
        # Calculate FID
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return float(fid)
    
    def calculate_psnr(self, img1, img2):
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return float(psnr)
    
    def calculate_ssim(self, img1, img2):
        """Calculate Structural Similarity Index"""
        # Convert to numpy and correct format
        img1_np = img1.cpu().numpy().transpose(1, 2, 0)
        img2_np = img2.cpu().numpy().transpose(1, 2, 0)
        
        # Ensure proper value range [0, 1]
        img1_np = np.clip(img1_np, 0, 1)
        img2_np = np.clip(img2_np, 0, 1)
        
        return ssim(img1_np, img2_np, multichannel=True, data_range=1.0)
    
    def measure_inference_time(self, model, input_tensor, num_runs=100):
        """Measure average inference time"""
        times = []
        model.eval()
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Actual timing runs
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(input_tensor)
                torch.cuda.synchronize()  # Ensure GPU operations are completed
                times.append(time.time() - start)
                
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times)
        }
    
    def measure_resource_usage(self, model, input_tensor):
        """Measure memory and CPU usage"""
        torch.cuda.reset_peak_memory_stats()
        
        # CPU Memory before
        memory_start = psutil.Process().memory_info().rss / 1024 / 1024
        
        # GPU Memory before
        gpu_memory_start = torch.cuda.memory_allocated() / 1024 / 1024
        
        # Run model
        with torch.no_grad():
            _ = model(input_tensor)
            torch.cuda.synchronize()
        
        # Memory after
        memory_end = psutil.Process().memory_info().rss / 1024 / 1024
        gpu_memory_end = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return {
            'cpu_memory_used_mb': memory_end - memory_start,
            'gpu_memory_used_mb': gpu_memory_end - gpu_memory_start,
            'gpu_memory_peak_mb': gpu_memory_peak,
            'cpu_percent': psutil.cpu_percent()
        }

    def evaluate_model_comprehensive(self, model, test_loader):
        """Comprehensive model evaluation"""
        results = {
            'fid_scores': [],
            'psnr_scores': [],
            'ssim_scores': [],
            'inference_times': [],
            'resource_usage': []
        }
        
        model.eval()
        with torch.no_grad():
            for content, style in test_loader:
                content = content.to(self.device)
                style = style.to(self.device)
                
                # Generate output
                output = model(content)
                
                # Calculate metrics
                results['fid_scores'].append(
                    self.calculate_fid(style, output)
                )
                results['psnr_scores'].append(
                    self.calculate_psnr(style, output)
                )
                results['ssim_scores'].append(
                    self.calculate_ssim(style[0], output[0])
                )
                
                # Performance metrics
                inference_time = self.measure_inference_time(
                    model, content, num_runs=10
                )
                results['inference_times'].append(inference_time)
                
                # Resource usage
                resource_usage = self.measure_resource_usage(model, content)
                results['resource_usage'].append(resource_usage)
        
        # Aggregate results
        final_results = {
            'quality_metrics': {
                'fid_mean': np.mean(results['fid_scores']),
                'fid_std': np.std(results['fid_scores']),
                'psnr_mean': np.mean(results['psnr_scores']),
                'psnr_std': np.std(results['psnr_scores']),
                'ssim_mean': np.mean(results['ssim_scores']),
                'ssim_std': np.std(results['ssim_scores'])
            },
            'performance_metrics': {
                'inference_time_mean': np.mean([t['mean_time'] for t in results['inference_times']]),
                'inference_time_std': np.std([t['mean_time'] for t in results['inference_times']])
            },
            'resource_metrics': {
                'cpu_memory_mean': np.mean([r['cpu_memory_used_mb'] for r in results['resource_usage']]),
                'gpu_memory_mean': np.mean([r['gpu_memory_used_mb'] for r in results['resource_usage']]),
                'gpu_memory_peak': max([r['gpu_memory_peak_mb'] for r in results['resource_usage']])
            }
        }
        
        return final_results
