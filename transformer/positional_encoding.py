import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embed_size):
        """

        :param max_seq_len: maximum token length of input sequence
        :param embed_size: embedding size
        """
        super(PositionalEncoding, self).__init__()
        self.max_seq_len = max_seq_len
        self.embed_size = embed_size

        pos_encoding = torch.zeros(max_seq_len, embed_size)
        for pos in range(max_seq_len):
            for i in range(0, embed_size, 2):
                pos_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embed_size)))
                pos_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / embed_size)))

        pos_encoding = pos_encoding.unsqueeze(0)  # for batch_size dimension
        # register_buffer is used to save tensor in model's state_dict, but it is not a trainable model parameter
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, seq_len):
        """
        generate positional encoding for sequence of word ids
        :param seq_len: shape (batch_size, seq_len)
        :return: shape (batch_size, seq_len, embed_size)
        """
        return self.pos_encoding[:, :seq_len, :]
