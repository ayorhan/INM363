# inference.py
import torch
from cnn_model import VGG19Features

def run_inference(input_image, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG19Features().to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    with torch.no_grad():
        _, style_feats = model(input_image.to(device))
    return style_feats
