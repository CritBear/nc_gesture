import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionEncoder(nn.Module):

    def __init__(self, max_seq_len, d_model, n_heads, n_layers):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        assert d_model % n_heads == 0, ValueError(
            f"( d_model % n_heads ) must be ( 0 )\n"
            f"d_model: {d_model}, n_heads: {n_heads}")

        self.self_attention_layers = nn.ModuleList([
            MultiHeadAttention(self.d_model, self.n_heads)
            for _ in range(self.n_layers)
        ])

    def forward(self, motion_feats):
        """
        :param motion_feats: (batch_size, feature_dim, max_seq_len) tensor
        :return: (batch_size, feature_dim, max_seq_len) tensor
        """

        for self_attention_layer in self.self_attention_layers:
            motion_attention = self_attention_layer(
                query=motion_feats,
                key=motion_feats,
                value=motion_feats
            )
            motion_feats = motion_feats + motion_attention

        return motion_feats


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()

        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.proj = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, kernel_size=1) for _ in range(3)]
        )
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.post_merge = nn.Sequential(
            nn.Conv1d(2 * d_model, 2 * d_model, kernel_size=1),
            nn.BatchNorm1d(2 * d_model),
            nn.ReLU(),
            nn.Conv1d(2 * d_model, d_model, kernel_size=1),
        )

        nn.init.constant_(self.post_merge[-1].bias, 0.0)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        _query, key, value = [l(x).view(batch_size, self.d_k, self.n_heads, -1) for l, x in
                             zip(self.proj, (query, key, value))]

        attention_scores = torch.einsum('bdhn,bdhm->bhnm', _query, key) / self.d_k ** .5
        attention_weight = torch.nn.functional.softmax(attention_scores, dim=-1)
        query_attention = torch.einsum('bhnm,bdhm->bdhn', attention_weight, value)

        query_attention = self.merge(query_attention.contiguous().view(batch_size, self.d_k * self.n_heads, -1))
        query_attention = self.post_merge(torch.cat([query_attention, query], dim=1))

        return query_attention
