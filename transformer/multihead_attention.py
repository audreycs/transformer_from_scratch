import torch.nn as nn
import torch
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert (self.head_dim * num_heads == embed_size), "wrong value for head_nums"

        self.Wq = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.Wk = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.Wv = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.Wo = nn.Linear(self.embed_size, self.embed_size, bias=True)

    def forward(self, query, key, value, mask=None):
        """
        generating multi-head attention from query, key, and value
        :param query: shape (batch_size, query_len, embed_size)
        :param key: shape (batch_size, seq_len, embed_size)
        :param value: shape (batch_size, seq_len, embed_size)
        :param mask: shape (batch_size, query_len, query_len)
        :return: shape (batch_size, query_len, embed_size)
        """
        batch_size = key.shape[0]
        seq_len = key.shape[1]
        query_len = query.shape[1]

        # split the embed_size into num_heads
        Q = query.view(batch_size, query_len, self.num_heads, self.head_dim)  # (batch_size, query_len, num_heads, head_dim)
        K = key.view(batch_size, seq_len, self.num_heads, self.head_dim)  # (batch_size, seq_len, num_heads, head_dim)
        V = value.view(batch_size, seq_len, self.num_heads, self.head_dim)  # (batch_size, seq_len, num_heads, head_dim)

        # linear projection
        Q = self.Wq(Q)  # (batch_size, query_len, num_heads, head_dim)
        K = self.Wk(K)  # (batch_size, seq_len, num_heads, head_dim)
        V = self.Wv(V)  # (batch_size, seq_len, num_heads, head_dim)

        # transpose to get dimensions batch_size, num_heads, seq_len, head_dim
        Q = Q.permute(0, 2, 1, 3)  # (batch_size, num_heads, query_len, head_dim)
        K = K.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

        QK = torch.matmul(Q, K.transpose(-2, -1))  # QK shape (batch_size, num_heads, query_len, seq_len)
        if mask is not None:
            # mask shape (batch_size, 1, seq_len, seq_len)
            QK = QK.masked_fill(mask == 0, float("-inf"))  # (batch_size, num_heads, query_len, seq_len)

        QK = QK / (self.head_dim ** 0.5)
        QK = F.softmax(QK, dim=-1)

        QKV = torch.matmul(QK, V)  # QKV shape (batch_size, num_heads, query_len, head_dim)
        QKV = QKV.permute(0, 2, 1, 3).contiguous()  # (batch_size, query_len, num_heads, head_dim)
        QKV = QKV.view(batch_size, query_len, self.embed_size)  # (batch_size, query_len, embed_size)
        attention = self.Wo(QKV)

        return attention
