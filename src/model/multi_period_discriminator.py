import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class PeriodicLayer(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        
        channels = [1, 32, 128, 512, 1024, 1024]
        
        self.conv_stack = nn.ModuleList()
        for i, (c_in, c_out) in enumerate(zip(channels[:-1], channels[1:])):
            current_stride = stride if i < 4 else 1
            self.conv_stack.append(
                weight_norm(nn.Conv2d(c_in, c_out, (kernel_size, 1), (current_stride, 1), padding=(2, 0)))
            )
            
        self.final_proj = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        feature_map = []
        b, c, t = x.shape
        
        remainder = t % self.period
        if remainder != 0:
            pad_size = self.period - remainder
            x = F.pad(x, (0, pad_size), mode="reflect")
            
        x = x.view(b, c, -1, self.period)
        
        for layer in self.conv_stack:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            feature_map.append(x)
            
        x = self.final_proj(x)
        feature_map.append(x)
        
        return torch.flatten(x, 1, -1), feature_map


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        periods = [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList([PeriodicLayer(p) for p in periods])

    def forward(self, real_audio, fake_audio, detach=False, **kwargs):
        if detach:
            fake_audio = fake_audio.detach()

        real_out = [disc(real_audio) for disc in self.discriminators]
        fake_out = [disc(fake_audio) for disc in self.discriminators]

        real_logits, real_feats = zip(*real_out)
        fake_logits, fake_feats = zip(*fake_out)

        return {
            'ans_tp': list(real_logits),
            'ans_fp': list(fake_logits),
            'feat_tp': list(real_feats),
            'feat_fp': list(fake_feats)
        }