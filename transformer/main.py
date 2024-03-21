import torch
from transformer import Transformer
from word_embedding import WordEmbedding
from positional_encoding import PositionalEncoding


def generate_target_mask(target):
    """
    generate mask for target sequence
    :param target: shape (batch_size, seq_len)
    :return: shape (batch_size, seq_len, seq_len)
    """
    batch_size, seq_len = target.size()
    mask = torch.tril(torch.ones(seq_len, seq_len)).expand(batch_size, 1, seq_len, seq_len)
    return mask


if __name__ == '__main__':
    # let 0 be sos token and 1 be eos token
    # src and target are already padded
    src = torch.tensor([[0, 2, 5, 6, 4, 3, 9, 5, 2, 9, 10, 1],
                        [0, 2, 8, 7, 3, 4, 5, 6, 7, 2, 10, 1]])
    target = torch.tensor([[0, 1, 7, 4, 3, 5, 9, 2, 8, 10, 9, 1],
                           [0, 1, 5, 6, 2, 4, 7, 6, 2, 8, 10, 1]])

    src_vocab_size = max(src.max(), target.max()) + 1
    seq_len = src.size(1)

    word_encoder = WordEmbedding(vocab_size=src_vocab_size, embed_size=64)
    position_encoder = PositionalEncoding(max_seq_len=16, embed_size=64)
    model = Transformer(embed_dim=64, num_heads=4, feedforward_dim=64, lay_nums=6, dropout=0.1)

    source_word_embedding = word_encoder(src)
    source_pos_embedding = position_encoder(seq_len)
    source_matrix = source_word_embedding + source_pos_embedding

    target_word_embedding = word_encoder(target)
    target_pos_embedding = position_encoder(seq_len)
    target_matrix = target_word_embedding + target_pos_embedding
    target_mask = generate_target_mask(target)

    out = model(source_matrix, target_matrix, target_mask)
    print(out)
