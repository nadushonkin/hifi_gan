import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


class SingleScaleDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(SingleScaleDiscriminator, self).__init__()
        
        norm_fn = spectral_norm if use_spectral_norm else weight_norm
        self.conv_modules = nn.ModuleList()
        
        # (in_ch, out_ch, kernel, stride, group, padding) для каждого слоя:
        layer_specs = [
            (1, 128, 15, 1, 1, 7),
            (128, 128, 41, 2, 4, 20),
            (128, 256, 41, 2, 16, 20),
            (256, 512, 41, 4, 16, 20),
            (512, 1024, 41, 4, 16, 20),
            (1024, 1024, 41, 1, 16, 20),
            (1024, 1024, 5, 1, 1, 2)
        ]

        for in_c, out_c, k, s, g, p in layer_specs:
            layer = norm_fn(nn.Conv1d(
                in_c, out_c, kernel_size=k, stride=s, groups=g, padding=p
            ))
            self.conv_modules.append(layer)

        self.classifier = norm_fn(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        feature_map_list = []
        out = x
        
        for layer in self.conv_modules:
            out = layer(out)
            out = F.leaky_relu(out, 0.1)
            feature_map_list.append(out)
            
        out = self.classifier(out)
        feature_map_list.append(out)
        
        return torch.flatten(out, 1), feature_map_list


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        
        self.sub_discriminators = nn.ModuleList([
            SingleScaleDiscriminator(use_spectral_norm=True),
            SingleScaleDiscriminator(use_spectral_norm=False),
            SingleScaleDiscriminator(use_spectral_norm=False)
        ])
        
        self.downsampler = nn.AvgPool1d(kernel_size=4, stride=2, padding=2)

    def _discriminator_pass(self, x, discriminator):
        return discriminator(x)

    def forward(self, real, fake, detach=False, **kwargs):
        if detach:
            fake = fake.detach()

        results = {
            'ans_ts': [], 'ans_fs': [],
            'feat_ts': [], 'feat_fs': []
        }

        current_real = real
        current_fake = fake

        for idx, disc_block in enumerate(self.sub_discriminators):
            if idx > 0:
                current_real = self.downsampler(current_real)
                current_fake = self.downsampler(current_fake)

            score_real, feats_real = self._discriminator_pass(current_real, disc_block)
            results['ans_ts'].append(score_real)
            results['feat_ts'].append(feats_real)

            score_fake, feats_fake = self._discriminator_pass(current_fake, disc_block)
            results['ans_fs'].append(score_fake)
            results['feat_fs'].append(feats_fake)

        return results