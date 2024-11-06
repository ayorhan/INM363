# train.py
import torch
import torch.optim as optim
import wandb
import argparse
from cnn_model import VGG19Features
from losses import StyleTransferLoss

def train_style_transfer(target_image, epochs, style_weight=1e6, content_weight=1e4):
    # Initialize Wandb
    wandb.init(project="cnn-style-transfer")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG19Features().to(device).eval()
    
    criterion = StyleTransferLoss(style_weight, content_weight)
    input_image = torch.randn_like(target_image, requires_grad=True)
    optimizer = optim.LBFGS([input_image])

    def closure():
        optimizer.zero_grad()
        content_feat, style_feats = model(input_image)
        target_content_feat, target_style_feats = model(target_image)
        loss = criterion(content_feat, style_feats, target_content_feat, target_style_feats)
        wandb.log({"Loss": loss.item()})
        loss.backward()
        return loss

    for epoch in range(epochs):
        optimizer.step(closure)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style Transfer Training")
    parser.add_argument("--nr_epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()
    
    # Placeholder for actual data loading
    target_image = torch.randn((1, 3, 256, 256))  # Replace with actual target image
    train_style_transfer(target_image, args.nr_epochs)
