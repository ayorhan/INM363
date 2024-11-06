# cnn_model.py
import torch
import torch.nn as nn
import torchvision.models as models

class VGG19Features(nn.Module):
    def __init__(self):
        super(VGG19Features, self).__init__()
        # Updated to use the new weights parameter instead of pretrained=True
        vgg19 = models.vgg19(weights="VGG19_Weights.DEFAULT").features
        self.style_layers = ['0', '5', '10', '19', '28']
        self.content_layers = ['21']
        self.model = nn.Sequential(*[vgg19[i] for i in range(29)])

    def forward(self, x):
        style_features = []
        content_feature = None
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.style_layers:
                style_features.append(x)
            if name in self.content_layers:
                content_feature = x
        return content_feature, style_features
