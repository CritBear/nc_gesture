#ACTOR의 코드를 가져옴

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class Encoder_TRANSFORMER(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_action_classes, num_label_classes,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_action_classes = num_action_classes
        self.num_label_classes = num_label_classes

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation # gelu를 사용.

        self.input_feats = self.njoints * self.nfeats # 26 * 6 = 156

        self.muQuery = nn.Parameter(torch.randn(self.num_label_classes, self.latent_dim)) # learnable token
        self.sigmaQuery = nn.Parameter(torch.randn(self.num_label_classes, self.latent_dim)) # learnable token

        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim) #linear projection

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

    def forward(self, batch):
        x, style, mask = batch["x"], batch["style"], batch["mask"]
        bs = x.shape[0]
        #bs, njoints, nfeats, nframes = x.shape
        #x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        x = x.permute(1, 0, 2)    # (bs, frames, joints * feats ) -> (frames, bs, joints * feats )

        # embedding of the skeleton
        x = self.skelEmbedding(x)

        # adding the mu and sigma queries
        xseq = torch.cat((self.muQuery[style][None],self.sigmaQuery[style][None], x), axis=0)

        # add positional encoding
        xseq = self.sequence_pos_encoder(xseq)

        # create a bigger mask, to allow attend to mu and sigma
        muandsigmaMask = torch.ones((bs, 2), dtype=bool, device=x.device)
        maskseq = torch.cat((muandsigmaMask, mask), axis=1)

        final = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)
        mu = final[0]
        logvar = final[1]

        return {"mu": mu, "logvar": logvar}


class Decoder_TRANSFORMER(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_action_classes,num_style_classes,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu",
                 ablation=None, **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints       #joint 수 = 26
        self.nfeats = nfeats         #feature 수 : rotation dimension = 6
        self.num_frames = num_frames #frame 수 (근데 안 쓰임)
        #self.num_action_classes = num_action_classes
        self.num_style_classes = num_style_classes #style(persona) 개수 = 4 (de,di,me,mi)

        self.latent_dim = latent_dim   #latent dimension = 64

        self.ff_size = ff_size         #dimension of feedforward network = 1024
        self.num_layers = num_layers   #decoder block 수 = 8
        self.num_heads = num_heads     #multi head attention의 head 수 = 4
        self.dropout = dropout         #dropout
        self.activation = activation   #activation function = gelu

        self.input_feats = self.njoints * self.nfeats #26*6 = 156

        self.styleBiases = nn.Parameter(torch.randn(self.num_style_classes, self.latent_dim))

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)

        self.finallayer = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, batch):
        z, style, mask = batch["z"], batch["style"], batch["mask"]

        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        njoints, nfeats = self.njoints, self.nfeats

        # shift the latent noise vector to be the action noise
        z = z + self.styleBiases[style]
        z = z[None]  # sequence of size 1

        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)

        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        output = self.finallayer(output).reshape(nframes, bs, njoints * nfeats)

        # zero for padded area
        output[~mask.T] = 0      #mask shape: (bs,nframe) , output : (nframes, bs)
        output = output.permute(1,0,2)

        batch["output"] = output
        return batch