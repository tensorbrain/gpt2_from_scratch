from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as f

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_emb: int = 384

class GPT(nn.module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transfomer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_emb),
            wpe = nn.Embedding(config.block_size, config.n_emb),
            h = nn.ModuleList([Block(config) for _ in range(config.layer)]),
            ln_f = nn.LayerNorm(config.n_emb),
        ))
        # Language Model clasiffier, projects embedding into a token class.
        self.lm_head = nn.Linear(config.n_emb, config.vocab_size, bias = false) 
