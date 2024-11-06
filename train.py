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
    print("Initialized Wandb")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG19Features().to(device).eval()
    print("Model loaded and moved to device:", device)
    
    criterion = StyleTransferLoss(style_weight, content_weight)
    input_image = torch.randn_like(target_image, requires_grad=True)
    optimizer = optim.LBFGS([input_image])
    print("Optimizer initialized")

    def closure():
        optimizer.zero_grad()
        print("Running model forward pass")
        content_feat, style_feats = model(input_image)
        print("Extracted features from input image")
        target_content_feat, target_style_feats = model(target_image)
        print("Extracted features from target image")
        
        loss = criterion(content_feat, style_feats, target_content_feat, target_style_feats)
        wandb.log({"Loss": loss.item()})
        print(f"Logged loss: {loss.item()}")

        loss.backward()
        print("Backward pass completed")
        return loss

    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}/{epochs}")
        optimizer.step(closure)
        print(f"Epoch {epoch+1} completed")
        
    print("Training completed")
    wandb.finish()
    print("Wandb run finished")

    torch.save(model.state_dict(), "cnn_style_transfer_checkpoint.pth")
print("Model checkpoint saved as cnn_style_transfer_checkpoint.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style Transfer Training")
    parser.add_argument("--nr_epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()
    
    # Placeholder for actual data loading
    target_image = torch.randn((1, 3, 128, 128))  # Reduced size for testing
    print("Starting training with reduced image size")
    train_style_transfer(target_image, args.nr_epochs)
