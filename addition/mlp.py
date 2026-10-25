"""Embedding MLP baseline for grokking modular addition."""
import torch
import torch.nn as nn
from data import get_args, run_experiment


class TinyMLP(nn.Module):
    def __init__(self, p, d=128, hidden=512):
        super().__init__()
        self.emb = nn.Embedding(p, d)
        self.net = nn.Sequential(
            nn.Linear(d * 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, p),
        )

    def forward(self, x):
        return self.net(self.emb(x).flatten(1))


if __name__ == '__main__':
    run_experiment('mlp', TinyMLP, get_args())
