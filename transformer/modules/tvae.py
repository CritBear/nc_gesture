import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from modules.encoder import Encoder
from modules.decoder import Decoder


class TVAE(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)


class TransformerEmbedding(nn.Module):

    def __init__(self, d_input, d_embed):
        super().__init__()

        self.d_input = d_input
        self.d_embed = d_embed

        self.pose_embedding = nn.Linear(self.d_iput, self.d_embed)
        self.positional_encoding = PositionalEncoding()

    def forward(self, x):
        out = self.pose_embedding(x) * math.sqrt(self.d_embed)
        out = self.positional_encoding(x)
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_embed, max_seq_len, device):
        super().__init__()
        encoding = torch.zeros(max_seq_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0).to(device)

    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out