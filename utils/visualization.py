import torch
from torchvision.utils import save_image
from pathlib import Path

def save_checkpoint_samples(model, val_loader, epoch, output_dir, model_type, config):
    """Generate and save sample outputs for the current checkpoint"""
    device = next(model.parameters()).device
    model.eval()
    
    # Create checkpoint-specific directory
    samples_dir = Path(output_dir) / 'training_progress' / f'epoch_{epoch}'
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        batch = next(iter(val_loader))
        content = batch['content'].to(device)
        style = batch['style'].to(device)
        
        num_samples = min(
            config.logging.visualization.num_samples,
            content.size(0)
        )
        
        if model_type.lower() == 'cyclegan':
            # Reference existing CycleGAN normalization from generate_style_transfer.py
            content_input = content * 2 - 1
            fake_B = model(content_input, direction='AB')
            
            if config.logging.visualization.save_cycle:
                fake_A = model(fake_B, direction='BA')
            
            if config.logging.visualization.save_identity:
                identity_B = model(style * 2 - 1, direction='AB')
            
            # Denormalize outputs
            fake_B = (fake_B + 1) * 0.5
            fake_A = (fake_A + 1) * 0.5 if config.logging.visualization.save_cycle else None
            identity_B = (identity_B + 1) * 0.5 if config.logging.visualization.save_identity else None
            
            for i in range(num_samples):
                images = [content[i], style[i], fake_B[i]]
                if fake_A is not None:
                    images.append(fake_A[i])
                if identity_B is not None:
                    images.append(identity_B[i])
                    
                comparison = torch.cat(images, dim=2)
                save_image(
                    comparison,
                    samples_dir / f'sample_{i}_cyclegan.png',
                    normalize=True
                )
        
        else:  # Johnson model
            output = model(content)
            
            for i in range(num_samples):
                images = [content[i], output[i]]
                if config.logging.visualization.save_style_targets:
                    images.insert(1, style[i])
                    
                comparison = torch.cat(images, dim=2)
                save_image(
                    comparison,
                    samples_dir / f'sample_{i}_johnson.png',
                    normalize=True
                )
    
    model.train()

def create_progress_visualization(output_dir, model_type):
    """Create an HTML file showing training progression"""
    progress_dir = Path(output_dir) / 'training_progress'
    epochs = sorted([d for d in progress_dir.iterdir() if d.is_dir()], 
                   key=lambda x: int(x.name.split('_')[1]))
    
    html_content = [
        '<!DOCTYPE html>',
        '<html>',
        '<head>',
        '<style>',
        '.sample { margin-bottom: 40px; }',
        '.epoch { margin-bottom: 20px; padding: 20px; border: 1px solid #ccc; }',
        '.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }',
        'img { max-width: 100%; height: auto; }',
        '</style>',
        '</head>',
        '<body>',
        f'<h1>{model_type} Training Progress</h1>'
    ]
    
    for epoch_dir in epochs:
        epoch_num = epoch_dir.name.split('_')[1]
        html_content.append(f'<div class="epoch"><h2>Epoch {epoch_num}</h2><div class="grid">')
        
        # Add all sample comparisons for this epoch
        samples = sorted(epoch_dir.glob(f'sample_*_{model_type.lower()}.png'))
        for sample in samples:
            html_content.append(
                f'<div class="sample">'
                f'<img src="{sample.relative_to(progress_dir)}" />'
                f'<p>Sample {sample.stem.split("_")[1]}</p>'
                f'</div>'
            )
        html_content.append('</div></div>')
    
    html_content.extend(['</body>', '</html>'])
    
    # Save HTML file
    with open(progress_dir / f'{model_type.lower()}_progress.html', 'w') as f:
        f.write('\n'.join(html_content))