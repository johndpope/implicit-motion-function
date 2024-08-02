import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import torchvision.models as models
from memory_profiler import profile
import colored_traceback.auto
# from vit import ImplicitMotionAlignment

DEBUG = False
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)



class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load a pre-trained ResNet model
        resnet = models.resnet50(pretrained=pretrained)
        
        # We'll use the first 4 layers of ResNet
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x):
        features = []
        x = self.layer0(x)
        features.append(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        return features


class DenseFeatureEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features



'''
The upsample parameter is replaced with downsample to match the diagram.
The first convolution now has a stride of 2 when downsampling.
The shortcut connection now uses a 3x3 convolution with stride 2 when downsampling, instead of a 1x1 convolution.
ReLU activations are applied both after adding the residual and at the end of the block.
The FeatResBlock is now a subclass of ResBlock with downsample=False, as it doesn't change the spatial dimensions.
'''
class TEResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if downsample else 1, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.downsample = downsample
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        debug_print(f"ResBlock input shape: {x.shape}")
        debug_print(f"ResBlock parameters: in_channels={self.in_channels}, out_channels={self.out_channels}, downsample={self.downsample}")

        residual = self.shortcut(x)
        debug_print(f"After shortcut: {residual.shape}")
        
        out = self.conv1(x)
        debug_print(f"After conv1: {out.shape}")
        out = self.bn1(out)
        out = self.relu1(out)
        debug_print(f"After bn1 and relu1: {out.shape}")
        
        out = self.conv2(out)
        debug_print(f"After conv2: {out.shape}")
        out = self.bn2(out)
        debug_print(f"After bn2: {out.shape}")
        
        out += residual
        debug_print(f"After adding residual: {out.shape}")
        
        out = self.relu2(out)
        debug_print(f"ResBlock output shape: {out.shape}")
        
        return out
        
class LatentTokenEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.resblock1 = TEResBlock(64, 64, downsample=False)
        self.resblock2 = TEResBlock(64, 128, downsample=True)
        self.resblock3 = TEResBlock(128, 256, downsample=True)
        
        self.resblock4 = nn.Sequential(
            TEResBlock(256, 512, downsample=True),
            TEResBlock(512, 512, downsample=False),
            TEResBlock(512, 512, downsample=False)
        )
        
        self.equal_conv = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        debug_print(f"💳 LatentTokenEncoder input shape: {x.shape}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        debug_print(f"    After first conv, bn, relu: {x.shape}")
        x = self.resblock1(x)
        debug_print(f"    After resblock1: {x.shape}")
        x = self.resblock2(x)
        debug_print(f"    After resblock2: {x.shape}")
        x = self.resblock3(x)
        debug_print(f"    After resblock3: {x.shape}")
        x = self.resblock4(x)
        debug_print(f"    After resblock4: {x.shape}")
        x = self.equal_conv(x)
        debug_print(f"    After equal_conv: {x.shape}")
        x = self.adaptive_pool(x)
        debug_print(f"    After adaptive_pool: {x.shape}")
        x = x.view(x.size(0), -1)
        debug_print(f"    After flatten: {x.shape}")
        t = self.fc_layers(x)
        debug_print(f"    1xdm=32 LatentTokenEncoder output shape: {t.shape}")
        return t


class LatentTokenDecoder(nn.Module):
    def __init__(self, latent_dim=32, base_channels=64, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.const = nn.Parameter(torch.randn(1, base_channels, 4, 4))
        self.fc = nn.Linear(latent_dim, base_channels)
        self.layers = nn.ModuleList()
        in_channels = base_channels
        for i in range(num_layers):
            out_channels = in_channels // 2 if i < num_layers - 1 else in_channels
            self.layers.append(
                StyleConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, style_dim=base_channels)
            )
            in_channels = out_channels

    def forward(self, x):
        style = self.fc(x)
        out = self.const.repeat(x.shape[0], 1, 1, 1)
        features = []
        for layer in self.layers:
            out = layer(out, style)
            out = F.relu(out)  # Apply ReLU activation after StyleConv2d
            features.append(out)
        return features
    
class StyleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, style_dim=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.modulation = nn.Linear(style_dim, in_channels)
        self.demodulation = True
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, style):
        batch, in_channel, height, width = x.shape
        style = self.modulation(style).view(batch, in_channel, 1, 1)
        x = x * style
        x = self.conv(x)
        if self.demodulation:
            demod = torch.rsqrt(x.pow(2).sum([2, 3], keepdim=True) + 1e-8)
            x = x * demod
        x = self.upsample(x)
        x = self.bn(x)
        return x





class ImplicitMotionAlignmentBROKEN(nn.Module):

    def __init__(self, feature_dim, motion_dim,layer_name, depth=2, heads=8, dim_head=64, mlp_dim=1024):
        super().__init__()
        self.feature_dim = feature_dim 
        self.motion_dim = motion_dim 
        self.num_heads = heads

        self.q_proj = nn.Linear(motion_dim, feature_dim)
        self.k_proj = nn.Linear(motion_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        self.cross_attention = nn.MultiheadAttention(feature_dim, heads)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        self.layer_name = layer_name
        debug_print(f"ImplicitMotionAlignment initialized: feature_dim={feature_dim}, motion_dim={motion_dim}")

    def forward(self, q, k, v):
        debug_print(f"🌏 ImplicitMotionAlignment input shapes -layer_name:{self.layer_name} q: {q.shape}, k: {k.shape}, v: {v.shape}")
        
        batch_size, c, h, w = v.shape

        # Reshape and project inputs
        q = self.q_proj(q.view(batch_size, self.motion_dim, -1).permute(0, 2, 1))
        k = self.k_proj(k.view(batch_size, self.motion_dim, -1).permute(0, 2, 1))
        v = self.v_proj(v.view(batch_size, self.feature_dim, -1).permute(0, 2, 1))

        debug_print(f"After projection - q: {q.shape}, k: {k.shape}, v: {v.shape}")

        # Ensure q, k, and v have the same sequence length
        seq_len = min(q.size(1), k.size(1), v.size(1))
        q = q[:, :seq_len, :]
        k = k[:, :seq_len, :]
        v = v[:, :seq_len, :]

        debug_print(f"After sequence length adjustment - q: {q.shape}, k: {k.shape}, v: {v.shape}")

        # Perform cross-attention
        attn_output, _ = self.cross_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        attn_output = attn_output.transpose(0, 1)
        debug_print(f"After cross-attention: {attn_output.shape}")
        
        x = self.norm1(q + attn_output)
        x = self.norm2(x + self.ffn(x))
        debug_print(f"After FFN: {x.shape}")

        # Reshape output to match original dimensions
        x = x.transpose(1, 2).contiguous()
        debug_print(f"After transpose: {x.shape}")
        debug_print(f"Attempting to reshape to: {(batch_size, self.feature_dim, h, w)}")
        
        # Check if the reshape is valid
        if x.numel() != batch_size * self.feature_dim * h * w:
            debug_print(f"WARNING: Cannot reshape tensor of size {x.numel()} into shape {(batch_size, self.feature_dim, h, w)}")
            # Adjust the output size to match the input size
            x = F.adaptive_avg_pool2d(x.view(batch_size, self.feature_dim, -1, 1), (h, w))
        else:
            x = x.view(batch_size, self.feature_dim, h, w)
        
        debug_print(f"Final output shape: {x.shape}")

        return x


class IMF(nn.Module):
    def __init__(self, latent_dim=32, base_channels=64, num_layers=4):
        super().__init__()
        self.dense_feature_encoder = ResNetFeatureExtractor()
        self.latent_token_encoder = LatentTokenEncoder(latent_dim=latent_dim)
        self.latent_token_decoder = LatentTokenDecoder(latent_dim=latent_dim, base_channels=base_channels, num_layers=num_layers)
        
        # Adjust feature_dims and motion_dims to match desired sizes
        self.feature_dims = [128, 256, 512, 1024]
        self.motion_dims = [256, 512, 512, 512]
        self.spatial_dims = [(64, 64), (32, 32), (16, 16), (8, 8)]
        
        self.implicit_motion_alignment = nn.ModuleList()
        for i in range(num_layers):
            feature_dim = self.feature_dims[i]
            motion_dim = self.motion_dims[i]
            spatial_dim = self.spatial_dims[i]
            alignment_module = ImplicitMotionAlignmentBROKEN(feature_dim=feature_dim, motion_dim=motion_dim,layer_name=i)
            # alignment_module = ImplicitMotionAlignment(feature_dim=feature_dim, motion_dim=motion_dim,spatial_dim=spatial_dim, layer_name=i)
            self.implicit_motion_alignment.append(alignment_module)

    def forward(self, x_current, x_reference):
        debug_print(f"IMF input shapes - x_current: {x_current.shape}, x_reference: {x_reference.shape}")

        f_r = self.dense_feature_encoder(x_reference)
        debug_print(f"Dense feature encoder output shapes: {[f.shape for f in f_r]}")

        t_r = self.latent_token_encoder(x_reference)
        t_c = self.latent_token_encoder(x_current)
        debug_print(f"Latent token shapes - t_r: {t_r.shape}, t_c: {t_c.shape}")

        m_r = self.latent_token_decoder(t_r)
        m_c = self.latent_token_decoder(t_c)
        debug_print(f"Latent token decoder output shapes - m_r: {[m.shape for m in m_r]}, m_c: {[m.shape for m in m_c]}")

        aligned_features = []
        for i, (f_r_i, m_r_i, m_c_i, align_layer) in enumerate(zip(f_r, m_r, m_c, self.implicit_motion_alignment)):
            debug_print(f"Layer {i} input shapes - f_r_i: {f_r_i.shape}, m_r_i: {m_r_i.shape}, m_c_i: {m_c_i.shape}")
            
            # Adjust feature dimensions
            f_r_i = F.interpolate(f_r_i, size=self.spatial_dims[i], mode='bilinear', align_corners=False)
            f_r_i = f_r_i[:, :self.feature_dims[i], :, :]
            
            # Adjust motion dimensions
            m_r_i = F.interpolate(m_r_i, size=self.spatial_dims[i], mode='bilinear', align_corners=False)
            m_r_i = torch.cat([m_r_i] * (self.motion_dims[i] // m_r_i.shape[1]), dim=1)
            m_c_i = F.interpolate(m_c_i, size=self.spatial_dims[i], mode='bilinear', align_corners=False)
            m_c_i = torch.cat([m_c_i] * (self.motion_dims[i] // m_c_i.shape[1]), dim=1)
            
            aligned_feature = align_layer(m_c_i, m_r_i, f_r_i)
            debug_print(f"Layer {i} aligned feature shape: {aligned_feature.shape}")
            aligned_features.append(aligned_feature)

        return aligned_features

    def process_tokens(self, t_c, t_r):
        debug_print(f"process_tokens input types - t_c: {type(t_c)}, t_r: {type(t_r)}")
        
        if isinstance(t_c, list) and isinstance(t_r, list):
            debug_print(f"process_tokens input shapes - t_c: {[tc.shape for tc in t_c]}, t_r: {[tr.shape for tr in t_r]}")
            m_c = [self.latent_token_decoder(tc) for tc in t_c]
            m_r = [self.latent_token_decoder(tr) for tr in t_r]
        else:
            debug_print(f"process_tokens input shapes - t_c: {t_c.shape}, t_r: {t_r.shape}")
            m_c = self.latent_token_decoder(t_c)
            m_r = self.latent_token_decoder(t_r)
        
        if isinstance(m_c[0], list):
            debug_print(f"process_tokens output shapes - m_c: {[[mc_i.shape for mc_i in mc] for mc in m_c]}, m_r: {[[mr_i.shape for mr_i in mr] for mr in m_r]}")
        else:
            debug_print(f"process_tokens output shapes - m_c: {[m.shape for m in m_c]}, m_r: {[m.shape for m in m_r]}")
        
        return m_c, m_r



class FrameDecoder(nn.Module):
    def __init__(self, feature_dims=[64, 256, 512, 1024]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(feature_dims) - 1):
            in_channels = feature_dims[-i - 1]
            out_channels = feature_dims[-i - 2]
            self.layers.append(
                ResBlock(in_channels, out_channels, upsample=True)
            )
        
        # Add one more upsampling layer to reach 512x512
        self.extra_upsample = ResBlock(feature_dims[0], feature_dims[0] // 2, upsample=True)
        
        self.final_conv = nn.Conv2d(feature_dims[0] // 2, 3, kernel_size=3, stride=1, padding=1)
        debug_print(f"FrameDecoder initialized with feature_dims: {feature_dims}")

    def forward(self, features):
        debug_print(f"🌻 FrameDecoder input features: {[f.shape for f in features]}")
        x = features[-1]
        debug_print(f"Starting x shape: {x.shape}")
        
        for i, layer in enumerate(self.layers):
            debug_print(f"Processing layer {i}")
            x = layer(x)
            debug_print(f"After layer {i}: {x.shape}")
            if i < len(self.layers) - 1:
                debug_print(f"Adding skip connection from features[-{i+2}]: {features[-i-2].shape}")
                x = x + F.interpolate(features[-i-2], size=x.shape[2:], mode='bilinear', align_corners=False)
                debug_print(f"After skip connection: {x.shape}")
        
        debug_print("Applying extra upsampling layer")
        x = self.extra_upsample(x)
        debug_print(f"After extra upsample: {x.shape}")
        
        debug_print("Applying final convolution")
        x = self.final_conv(x)
        debug_print(f"After final conv: {x.shape}")
        
        debug_print("Applying tanh activation")
        x = torch.tanh(x)
        debug_print(f"Final output shape: {x.shape}")
        
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else nn.Identity()
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        debug_print(f"ResBlock initialized with in_channels: {in_channels}, out_channels: {out_channels}, upsample: {upsample}")

    def forward(self, x):
        debug_print(f"ResBlock input shape: {x.shape}")
        residual = self.shortcut(x)
        debug_print(f"ResBlock shortcut shape: {residual.shape}")
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        debug_print(f"ResBlock before adding residual: {out.shape}")
        out += residual
        out = self.relu(out)
        out = self.upsample(out)
        debug_print(f"ResBlock output shape: {out.shape}")
        return out
    

class IMFModel(nn.Module):
    def __init__(self, latent_dim=32, base_channels=64, num_layers=4):
        super().__init__()
        self.imf = IMF(latent_dim, base_channels, num_layers)
        self.frame_decoder = FrameDecoder(feature_dims=[64, 256, 512, 1024])

    def forward(self, x_current, x_reference):
        x_current = x_current.requires_grad_()
        x_reference = x_reference.requires_grad_()
        aligned_features = self.imf(x_current, x_reference)
        reconstructed_frame = self.frame_decoder(aligned_features)
        if self.training:
            grads = torch.autograd.grad(reconstructed_frame.sum(), [x_current, x_reference], retain_graph=True, allow_unused=True)
        return reconstructed_frame

    @property
    def latent_token_encoder(self):
        return self.imf.latent_token_encoder

    @property
    def dense_feature_encoder(self):
        return self.imf.dense_feature_encoder

    @property
    def latent_token_decoder(self):
        return self.imf.latent_token_decoder