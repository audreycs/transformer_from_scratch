import torch.nn as nn
import torch
from multihead_attention import MultiHeadAttention


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout

        self.multihead_attention = MultiHeadAttention(embed_dim, num_heads)

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(self.dropout)

        self.forward_layer = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim)
        )

        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(self.dropout)

    def forward(self, query, key, value):
        attention = self.multihead_attention(query, key, value, mask=None)
        # residual connection and layer normalization
        attention = self.layer_norm1(attention + query)
        # apply dropout
        attention = self.dropout1(attention)

        forward = self.forward_layer(attention)
        # residual connection and layer normalization
        forward = self.layer_norm2(forward + attention)
        # apply dropout
        forward = self.dropout2(forward)

        return forward
