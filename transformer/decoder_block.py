import torch.nn as nn
from multihead_attention import MultiHeadAttention
from encoder_block import EncoderBlock


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, feedforward_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout

        self.masked_multihead_attention = MultiHeadAttention(embed_size, num_heads)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.attention2 = EncoderBlock(embed_size, num_heads, feedforward_dim, dropout)

    def forward(self, query, key, value, mask):
        masked_attention = self.masked_multihead_attention(query=query, key=query, value=query, mask=mask)
        # residual connection and layer normalization
        masked_attention = self.layer_norm(masked_attention + query)
        # apply dropout
        masked_attention = self.dropout(masked_attention)

        attention = self.attention2(query=masked_attention, key=key, value=value)

        return attention
