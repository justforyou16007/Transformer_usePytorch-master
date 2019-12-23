import torch.nn as nn

from .transformer_block import TransformerBlock
from .TokenTensorizer import TokenTensorizer


class BERT(nn.Module):

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.feed_forward_hidden = hidden * 4

        self.embedding = TokenTensorizer(vocab_size=vocab_size, embed_size=hidden)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):

        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(x, segment_info)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
