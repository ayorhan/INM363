import argparse
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import yaml

from models import get_model
from utils.config import StyleTransferConfig

def load_image(image_path: str, image_size: int = 256) -> torch.Tensor:
    """Load and preprocess image"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def create_comparison_grid(content_image: torch.Tensor, 
                         style_image: torch.Tensor, 
                         output_image: torch.Tensor,
                         save_path: str):
    """Create a side-by-side comparison of content, style, and output images"""
    # Ensure all images are on CPU and denormalized
    def denormalize(img):
        return torch.clamp(img * 0.5 + 0.5, 0, 1)
    
    content = denormalize(content_image.cpu())
    style = denormalize(style_image.cpu())
    output = denormalize(output_image.cpu())
    
    # Create the comparison grid (1 row, 3 columns)
    comparison = torch.cat([content, style, output], dim=2)
    
    # Add labels
    from PIL import Image, ImageDraw, ImageFont
    img_pil = transforms.ToPILImage()(comparison)
    draw = ImageDraw.Draw(img_pil)
    
    # Calculate text positions (centered under each image)
    img_width = img_pil.size[0]
    img_height = img_pil.size[1]
    section_width = img_width // 3
    y_position = img_height - 30
    
    # Add labels
    labels = ['Content', 'Style', 'Output']
    for i, label in enumerate(labels):
        x_position = (i * section_width) + (section_width // 2)
        draw.text((x_position, y_position), label, 
                 fill='white', anchor='mm')
    
    # Save the comparison
    img_pil.save(save_path)

def generate_styled_image(
    config_path: str,
    checkpoint_path: str,
    content_path: str,
    style_path: str,
    output_path: str,
    device: str = 'cuda'
):
    """Generate styled image using trained model"""
    print("\n=== Starting Style Transfer Process ===")
    
    # Check if CUDA is available when device is set to 'cuda'
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        device = 'cpu'
    print(f"Using device: {device}")
    
    # Load configuration
    print("\nLoading configuration...")
    config = StyleTransferConfig(config_path)
    print(f"Model type: {config.model.model_type}")
    print(f"Image size: {config.data.image_size}x{config.data.image_size}")
    
    # Initialize model
    print("\nInitializing model...")
    model = get_model(config)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    # Print model status after loading
    print(f"Model parameters loaded: {sum(p.numel() for p in model.parameters())}")
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Load images
    print("\nLoading and preprocessing images...")
    print(f"Content image: {content_path}")
    content_image = load_image(content_path, config.data.image_size).to(device)
    print(f"Style image: {style_path}")
    style_image = load_image(style_path, config.data.image_size).to(device)
    
    # Print input image ranges
    print(f"Content image range: {content_image.min().item():.3f} to {content_image.max().item():.3f}")
    
    # Generate output
    print("\nGenerating styled image...")
    with torch.no_grad():
        if config.model.model_type.lower() == 'johnson':
            # Johnson model takes only content image as input
            output = model(content_image)
            print("Using Johnson model inference")
            print(f"Output tensor range: {output.min().item():.3f} to {output.max().item():.3f}")
        elif config.model.model_type.lower() == 'cyclegan':
            # CycleGAN takes only content image as input to G_AB
            # Ensure input is in correct range [-1, 1]
            content_input = content_image * 2 - 1
            print(f"Input tensor range to CycleGAN: {content_input.min().item():.3f} to {content_input.max().item():.3f}")
            
            output = model(content_input, direction='AB')
            print("Using CycleGAN generator")
            print(f"Raw output tensor range: {output.min().item():.3f} to {output.max().item():.3f}")
            
            # Direct normalization to [0, 1] range for saving
            output = (output + 1) * 0.5
            print(f"Normalized output tensor range: {output.min().item():.3f} to {output.max().item():.3f}")
    
    # Save individual output
    print("\nSaving images...")
    output_image = output.cpu().squeeze(0)
    
    # Ensure output is in [0, 1] range
    output_image = torch.clamp(output_image, 0, 1)
    print(f"Final output range: {output_image.min().item():.3f} to {output_image.max().item():.3f}")
    
    save_image(output_image, output_path)
    
    # Create and save comparison
    comparison_path = output_path.rsplit('.', 1)[0] + '_comparison.png'
    create_comparison_grid(content_image.squeeze(0), 
                         style_image.squeeze(0), 
                         output_image,
                         comparison_path)
    
    print(f"\n✓ Style transfer complete!")
    print(f"✓ Output saved to: {output_path}")
    print(f"✓ Comparison saved to: {comparison_path}")
    print("\n=== Process Finished ===")

def main():
    parser = argparse.ArgumentParser(description='Generate styled image using trained model')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to model configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--content', type=str, required=True,
                      help='Path to content image')
    parser.add_argument('--style', type=str, required=True,
                      help='Path to style image')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to save output image')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    generate_styled_image(
        args.config,
        args.checkpoint,
        args.content,
        args.style,
        args.output,
        args.device
    )

if __name__ == "__main__":
    main() 