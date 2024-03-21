import torch.nn as nn


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        """

        :param vocab_size: vocabulary size
        :param embed_size: embedding size
        """
        super(WordEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, seq_ids):
        """
        generate word embedding from sequence of word ids
        :param seq_ids: shape (batch_size, seq_len)
        :return: shape (batch_size, seq_len, embed_size)
        """
        embeddings = self.embedding(seq_ids)
        return embeddings
