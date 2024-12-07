import numpy as np

class ModelComparator:
    def __init__(self, models_dict, evaluator):
        self.models = models_dict  # Dictionary of model_name: model_instance
        self.evaluator = evaluator
        self.results = {}
    
    def run_comparison(self, test_dataset):
        """Run comprehensive comparison of all models"""
        for name, model in self.models.items():
            self.results[name] = {
                'quality_metrics': self._evaluate_quality(model, test_dataset),
                'performance_metrics': self._evaluate_performance(model, test_dataset),
                'resource_metrics': self._evaluate_resources(model, test_dataset)
            }
        return self.results
    
    def _evaluate_quality(self, model, dataset):
        """Evaluate image quality metrics"""
        fid_scores = []
        psnr_scores = []
        ssim_scores = []
        
        for content, style in dataset:
            output = model(content)
            fid_scores.append(self.evaluator.calculate_fid(style, output))
            psnr_scores.append(self.evaluator.calculate_psnr(style, output))
            ssim_scores.append(self.evaluator.calculate_ssim(style[0], output[0]))
        
        return {
            'fid': np.mean(fid_scores),
            'psnr': np.mean(psnr_scores),
            'ssim': np.mean(ssim_scores)
        }
    
    def _evaluate_performance(self, model, dataset):
        """Evaluate performance metrics"""
        sample_input = next(iter(dataset))[0]
        return {
            'inference_time': self.evaluator.measure_inference_time(model, sample_input),
            'model_size': sum(p.numel() for p in model.parameters())
        }
    
    def _evaluate_resources(self, model, dataset):
        """Evaluate resource usage"""
        sample_input = next(iter(dataset))[0]
        return self.evaluator.measure_resource_usage(model, sample_input)
