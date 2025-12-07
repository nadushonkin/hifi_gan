import torch
from torch import nn


def calculate_discriminator_loss(real_outputs, fake_outputs):
    total_loss = sum(
        torch.mean(torch.pow(torch.sub(1.0, t), 2)) + 
        torch.mean(torch.pow(f, 2))
        for t, f in zip(real_outputs, fake_outputs)
    )
    
    return total_loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mpd_real_out, mpd_fake_out, msd_real_out, msd_fake_out, **batch_info):
        mpd_loss_component = calculate_discriminator_loss(mpd_real_out, mpd_fake_out)
        msd_loss_component = calculate_discriminator_loss(msd_real_out, msd_fake_out)
        
        total_loss = mpd_loss_component + msd_loss_component
        
        return {
            "loss": total_loss,
            "mpd_loss": mpd_loss_component,
            "msd_loss": msd_loss_component
        }
    