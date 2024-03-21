import torch.nn as nn
from encoder_block import EncoderBlock
from decoder_block import DecoderBlock


class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, lay_nums, dropout=0.1):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.lay_nums = lay_nums
        self.dropout = dropout

        self.encoder = nn.ModuleList([EncoderBlock(embed_dim, num_heads, feedforward_dim, dropout) for _ in range(lay_nums)])
        self.decoder = nn.ModuleList([DecoderBlock(embed_dim, num_heads, feedforward_dim, dropout) for _ in range(lay_nums)])

        self.linear = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, source_matrix, target_matrix, target_mask):
        x = source_matrix
        for i in range(self.lay_nums):
            x = self.encoder[i](query=x, key=x, value=x)

        encoder_output = x
        x = target_matrix
        for i in range(self.lay_nums):
            x = self.decoder[i](query=x, key=encoder_output, value=encoder_output, mask=target_mask)

        x = self.linear(x)
        out = self.softmax(x)

        return out
