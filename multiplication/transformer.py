"""Causal, one-block Transformer for grokking modular multiplication."""
import torch
import torch.nn as nn
from data import get_args, run_experiment


class TinyTransformer(nn.Module):
    def __init__(self, p, d=128, heads=4, ff=512):
        super().__init__()
        self.p = p
        self.emb = nn.Embedding(p + 1, d)  # p is the '=' token
        self.pos = nn.Parameter(torch.randn(3, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.out = nn.Linear(d, p)
        self.register_buffer('mask', torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1))

    def forward(self, x):
        eq = torch.full((x.size(0), 1), self.p, dtype=torch.long, device=x.device)
        tokens = torch.cat((x, eq), dim=1)
        h = self.emb(tokens) + self.pos
        att, _ = self.attn(h, h, h, attn_mask=self.mask, need_weights=False)
        h = h + att
        h = h + self.ffn(h)
        return self.out(h[:, -1])


if __name__ == '__main__':
    run_experiment('transformer', TinyTransformer, get_args())
