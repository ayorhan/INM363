# inference.py
import torch
from cnn_model import VGG19Features
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# Function to load and preprocess the input image
def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    
    in_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Preprocess image
    image = in_transform(image).unsqueeze(0)  # Add batch dimension
    print("Loaded and transformed image")
    return image

# Function to run inference
def run_inference(input_image_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device set to:", device)
    
    # Load model and checkpoint
    model = VGG19Features().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("Loaded model and checkpoint")

    # Load and preprocess input image
    input_image = load_image(input_image_path).to(device)
    print("Image loaded to device")

    # Run inference to extract style features
    with torch.no_grad():
        _, style_feats = model(input_image)
    print("Style features extracted")

    # Display the input image (as an example)
    input_image_display = input_image.squeeze().cpu().permute(1, 2, 0).numpy()
    plt.imshow(input_image_display)
    plt.axis("off")
    plt.show()
    print("Displayed image")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style Transfer Inference")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")

    args = parser.parse_args()
    run_inference(args.input_image, args.checkpoint)
