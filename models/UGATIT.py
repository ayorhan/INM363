"""
U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization

Source: https://arxiv.org/abs/1907.10830
Type: GAN (Generative Adversarial Network)

Architecture:
- Attention-guided generators
- Adaptive layer-instance normalization
- CAM-based attention module
- Multi-scale discriminators
- Auxiliary classifier

Pros:
- Unsupervised learning
- Attention-guided translation
- Good geometric changes
- Stable training

Cons:
- High memory requirements
- Long training time
- Complex architecture
- Parameter sensitivity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class AdaILN(nn.Module):
    """Adaptive Instance-Layer Normalization"""
    def __init__(self, num_features, eps=1e-5):
        super(AdaILN, self).__init__()
        self.eps = eps
        self.rho = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=6):
        super(Generator, self).__init__()
        
        # Encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                nn.InstanceNorm2d(ngf),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf*2),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf*4),
                nn.ReLU(True)
            )
        ])
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([ResnetBlock(ngf*4) for _ in range(n_blocks)])
        
        # CAM
        self.gap_fc = nn.Linear(ngf*4, 1, bias=False)
        self.gmp_fc = nn.Linear(ngf*4, 1, bias=False)
        self.conv1x1 = nn.Conv2d(ngf*8, ngf*4, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)
        
        # Gamma, Beta
        self.fc = nn.Sequential(
            nn.Linear(ngf*4, ngf*4),
            nn.ReLU(True),
            nn.Linear(ngf*4, ngf*4),
            nn.ReLU(True)
        )
        
        # Decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf*2),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                nn.Tanh()
            )
        ])
        
        # AdaILN
        self.ada_iln_blocks = nn.ModuleList([AdaILN(ngf*4) for _ in range(n_blocks)])

    def forward(self, input):
        # Encoding
        x = input
        encoder_features = []
        for block in self.encoder:
            x = block(x)
            encoder_features.append(x)
        
        # CAM
        gap = F.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)
        
        gmp = F.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        
        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))
        
        # Compute gamma and beta
        heatmap = torch.sum(x, dim=1, keepdim=True)
        if self.training:
            mean = heatmap.view(heatmap.size(0), -1).mean(dim=1).view(heatmap.size(0), 1, 1, 1)
            var = heatmap.view(heatmap.size(0), -1).var(dim=1).view(heatmap.size(0), 1, 1, 1)
            heatmap = (heatmap - mean) / var
            
        feature = self.fc(F.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1))
        gamma, beta = feature.chunk(2, 1)
        
        # Bottleneck with AdaILN
        for block, ada_iln in zip(self.bottleneck, self.ada_iln_blocks):
            x = block(x)
            x = ada_iln(x, gamma, beta)
        
        # Decoding
        for block in self.decoder:
            x = block(x)
        
        return x, cam_logit, heatmap

class CAMBlock(nn.Module):
    def __init__(self, in_channels):
        super(CAMBlock, self).__init__()
        self.gap_fc = nn.Linear(in_channels, 1, bias=False)
        self.gmp_fc = nn.Linear(in_channels, 1, bias=False)
        self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        gap = F.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = F.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))
        
        heatmap = torch.sum(x, dim=1, keepdim=True)
        return x, cam_logit, heatmap

class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True)
        )
        
        self.cam = CAMBlock(ndf*8)
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        out, cam_logit, heatmap = self.cam(x)
        return out, cam_logit, heatmap

class UGATIT(nn.Module):
    def __init__(self, config):
        super(UGATIT, self).__init__()
        self.config = config
        
        # Generators
        self.genA2B = Generator(config['input_nc'], config['output_nc'])
        self.genB2A = Generator(config['output_nc'], config['input_nc'])
        
        # Discriminators
        self.disGA = Discriminator(config['input_nc'])
        self.disGB = Discriminator(config['output_nc'])
        self.disLA = Discriminator(config['input_nc'])
        self.disLB = Discriminator(config['output_nc'])
        
        # Initialize optimizers
        self.G_optim = torch.optim.Adam(
            list(self.genA2B.parameters()) + list(self.genB2A.parameters()),
            lr=config['lr'],
            betas=(0.5, 0.999)
        )
        self.D_optim = torch.optim.Adam(
            list(self.disGA.parameters()) + list(self.disGB.parameters()) +
            list(self.disLA.parameters()) + list(self.disLB.parameters()),
            lr=config['lr'],
            betas=(0.5, 0.999)
        )
        
    def train_step(self, real_A, real_B):
        self.G_optim.zero_grad()
        self.D_optim.zero_grad()
        
        # Generate fake images
        fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
        fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)
        
        # Cycle
        fake_A2B2A, _, _ = self.genB2A(fake_A2B)
        fake_B2A2B, _, _ = self.genA2B(fake_B2A)
        
        # Discriminator outputs
        real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
        real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
        real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
        real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)
        
        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)
        
        # Compute losses
        G_adv_loss = self.compute_generator_loss(
            fake_GA_logit, fake_GA_cam_logit,
            fake_GB_logit, fake_GB_cam_logit,
            fake_LA_logit, fake_LA_cam_logit,
            fake_LB_logit, fake_LB_cam_logit
        )
        
        G_cycle_loss = self.compute_cycle_loss(
            real_A, fake_A2B2A,
            real_B, fake_B2A2B
        )
        
        G_cam_loss = self.compute_cam_loss(
            fake_A2B_cam_logit,
            fake_B2A_cam_logit
        )
        
        G_loss = G_adv_loss + G_cycle_loss + G_cam_loss
        
        # Update networks
        G_loss.backward()
        self.G_optim.step()
        
        D_loss = self.compute_discriminator_loss(
            real_GA_logit, fake_GA_logit,
            real_GB_logit, fake_GB_logit,
            real_LA_logit, fake_LA_logit,
            real_LB_logit, fake_LB_logit
        )
        
        D_loss.backward()
        self.D_optim.step()
        
        return {
            'G_loss': G_loss.item(),
            'D_loss': D_loss.item(),
            'G_adv_loss': G_adv_loss.item(),
            'G_cycle_loss': G_cycle_loss.item(),
            'G_cam_loss': G_cam_loss.item()
        } 