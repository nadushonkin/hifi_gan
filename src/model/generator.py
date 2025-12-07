import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        m.weight.data.normal_(0.0, 0.01)

class ResBlock_V3(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=(1, 3)):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1),
                weight_norm(nn.Conv1d(
                    channels, 
                    channels, 
                    kernel_size, 
                    stride=1, 
                    dilation=d,
                    padding=(kernel_size - 1) * d // 2
                ))
            ) for d in dilations
        ])
        
        self.blocks.apply(init_weights)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


class MRF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, res_kernels, res_dilations):
        super().__init__()
        self.upsample = weight_norm(nn.ConvTranspose1d(
            in_channels, 
            out_channels, 
            kernel_size,
            stride=stride,
            padding=(kernel_size - stride) // 2
        ))
        
        self.res_blocks = nn.ModuleList([
            ResBlock_V3(out_channels, k, d) 
            for k, d in zip(res_kernels, res_dilations)
        ])
        
        self.upsample.apply(init_weights)
        self.res_blocks.apply(init_weights)

    def forward(self, x):
        x = F.leaky_relu(x, 0.1)
        x_up = self.upsample(x)
        
        outputs = [block(x_up) for block in self.res_blocks]
        return sum(outputs) / len(outputs)


class Generator(nn.Module):
    def __init__(self, input_channels, base_channels, kernel_sizes, res_kernels, res_dilations):
        super().__init__()
        
        self.pre_conv = weight_norm(nn.Conv1d(input_channels, base_channels, 7, 1, padding=3))

        blocks = []
        for i, k in enumerate(kernel_sizes):
            ch_in = base_channels // (2 ** i)
            ch_out = base_channels // (2 ** (i + 1))
            
            blocks.append(MRF(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=k,
                stride=k // 2,
                res_kernels=res_kernels,
                res_dilations=res_dilations
            ))
            
        self.main_body = nn.Sequential(*blocks)
        
        last_channels = base_channels // (2 ** len(kernel_sizes))
        self.post_conv = weight_norm(nn.Conv1d(last_channels, 1, 7, 1, padding=3))

        self.pre_conv.apply(init_weights)
        self.post_conv.apply(init_weights)

    def forward(self, mel_input, **kwargs):
        x = self.pre_conv(mel_input)
        x = self.main_body(x)
        
        x = F.leaky_relu(x)
        x = self.post_conv(x)
        x = torch.tanh(x)
        
        return {'fake': x}