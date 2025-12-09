import torch
from torch import nn
import torch.nn.functional as F


def gen_loss(disc_outputs):
    return sum(torch.mean(torch.square(torch.sub(1.0, out))) for out in disc_outputs)


def feat_loss(real_features, fake_features):
    loss_sum = 0.0
    for ft, ff in zip(real_features, fake_features):
        for t, f in zip(ft, ff):
            if t.shape != f.shape:
                if len(t.shape) == 3:  # Conv1d: [B, C, T]
                    min_size = min(t.shape[2], f.shape[2])
                    t = t[..., :min_size]
                    f = f[..., :min_size]
                elif len(t.shape) == 4:  # Conv2d: [B, C, H, W]
                    min_h = min(t.shape[2], f.shape[2])
                    min_w = min(t.shape[3], f.shape[3])
                    t = t[:, :, :min_h, :min_w]
                    f = f[:, :, :min_h, :min_w]
            loss_sum += torch.mean(torch.abs(f - t))
    
    return 2.0 * loss_sum


class GeneratorLoss(nn.Module):
    def __init__(self, l1_weight=45.0):
        super().__init__()
        self._l1_weight = l1_weight

    def forward(self, ans_fp, ans_fs, feat_tp, feat_fp, feat_ts, feat_fs,
                real_mel, fake_mel, **batch):
        g_loss_component = gen_loss(ans_fp) + gen_loss(ans_fs)
        f_loss_component = feat_loss(feat_tp, feat_fp) + feat_loss(feat_ts, feat_fs)
        
        if real_mel.shape != fake_mel.shape:
            # Mel spectrograms shape: [B, n_mels, T]
            min_time = min(real_mel.shape[2], fake_mel.shape[2])
            real_mel = real_mel[:, :, :min_time]
            fake_mel = fake_mel[:, :, :min_time]
        
        l1_loss_component = F.l1_loss(fake_mel, real_mel) * self._l1_weight
        
        total_loss = g_loss_component + f_loss_component + l1_loss_component
        
        return {
            "loss": total_loss, 
            "g_loss": g_loss_component, 
            "feat_loss": f_loss_component,
            "l1_loss": l1_loss_component
        }