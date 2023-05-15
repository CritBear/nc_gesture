import torch
import torch.nn as nn
from torch import autograd
import numpy as np
import torch.nn.functional as F
from nc_gesture import nc_attention_encoder as AE
from nc_gesture.style_transfer.modules.transformer import *

def kl_divergence(mu, logvar):
    # mu: 평균 벡터, logvar: 로그 분산 벡터
    # KL divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_loss

def compute_mse_loss(batch):
    x = batch['target_motion']
    output = batch['output']
    mask = batch['mask']
    gtmasked = x[mask]
    outmasked = output[mask]
    loss = F.mse_loss(gtmasked, outmasked, reduction='mean')
    return loss

def compute_vel_mse_loss(batch):
    x = batch["output"].permute(0, 2, 1)
    output = batch["x"].permute(0, 2, 1)

    gtvel = (x[..., 1:] - x[..., :-1])
    outputvel = (output[..., 1:] - output[..., :-1])

    mask = batch["mask"][..., 1:]

    gtvelmasked = gtvel.permute(0, 2, 1)[mask]
    outvelmasked = outputvel.permute(0, 2,1)[mask]

    loss = F.mse_loss(gtvelmasked, outvelmasked, reduction='mean')
    return loss
def compute_avg_vel_mse_loss(batch):
    x = batch["output"].permute(0, 2, 1)
    output = batch["target_motion"].permute(0, 2, 1)

    gtvel = (x[..., 1:] - x[..., :-1])
    outputvel = (output[..., 1:] - output[..., :-1])

    mask = batch["mask"][..., 1:]

    gtvelmasked = gtvel.permute(0, 2, 1)[mask]
    outvelmasked = outputvel.permute(0, 2,1)[mask]

    gtvelavg = torch.mean(torch.abs(gtvelmasked), dim=0)
    outvelavg = torch.mean(torch.abs(outvelmasked), dim=0)

    loss = F.mse_loss(gtvelavg, outvelavg, reduction='mean')
    return loss

class TVAE(nn.Module):
    def __init__(self, config):
        super(TVAE, self).__init__()
        self.config = config
        self.encoder = Encoder_TRANSFORMER("TVAE",config.num_joints,config.num_feats,config.num_channel,
                                           config.num_action_classes,config.num_style_classes,latent_dim= config.latent_dim,num_heads=config.num_heads)
        self.decoder = Decoder_TRANSFORMER("TVAE",config.num_joints,config.num_feats,config.num_channel,
                                           config.num_action_classes,config.num_style_classes,latent_dim= config.latent_dim,num_heads=config.num_heads)
    def reparameterize(self, batch, seed=None):
        mu, logvar = batch["mu"], batch["logvar"]
        std = torch.exp(logvar / 2)

        if seed is None:
            eps = std.data.new(std.size()).normal_()
        else:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)

        z = eps.mul(std).add_(mu)
        return z

    def forward(self, batch):
        #encode
        batch.update(self.encoder(batch))
        batch["z"] = self.reparameterize(batch)

        # decode
        batch.update(self.decoder(batch))

        return batch
