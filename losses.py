# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def gram_matrix(input_tensor):
    _, d, h, w = input_tensor.size()
    features = input_tensor.view(d, h * w)
    G = torch.mm(features, features.t())
    return G.div(d * h * w)

class StyleTransferLoss(nn.Module):
    def __init__(self, style_weight, content_weight):
        super(StyleTransferLoss, self).__init__()
        self.style_weight = style_weight
        self.content_weight = content_weight

    def forward(self, content_feat, style_feats, target_content, target_styles):
        content_loss = F.mse_loss(content_feat, target_content)
        style_loss = sum(F.mse_loss(gram_matrix(style), gram_matrix(target))
                         for style, target in zip(style_feats, target_styles))
        return self.content_weight * content_loss + self.style_weight * style_loss
