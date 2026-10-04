"""~6M-param decoder-only transformer for action-value prediction.

Input: 77 FEN tokens + 1 action token (78 total). Output: logits over
NUM_BINS win-probability bins, read at the final position.
Post-norm + SwiGLU + learned positional embeddings, per the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import common

D_MODEL = 256
N_LAYERS = 7
N_HEADS = 8
FFN_HIDDEN = 688
CTX = common.SEQ_LEN + 1  # 78


HEAD_DIM = D_MODEL // N_HEADS


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.w_gate = nn.Linear(D_MODEL, FFN_HIDDEN, bias=False)
        self.w_up = nn.Linear(D_MODEL, FFN_HIDDEN, bias=False)
        self.w_down = nn.Linear(FFN_HIDDEN, D_MODEL, bias=False)
        self.norm2 = nn.LayerNorm(D_MODEL)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, N_HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = self.norm1(x + self.proj(a.transpose(1, 2).reshape(b, t, D_MODEL)))
        h = self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        return self.norm2(x + h)


class ActionValueTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(common.VOCAB_SIZE, D_MODEL)
        self.act_emb = nn.Embedding(common.NUM_ACTIONS, D_MODEL)
        self.pos_emb = nn.Embedding(CTX, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYERS))
        self.head = nn.Linear(D_MODEL, common.NUM_BINS, bias=False)

    def forward(self, fen_tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """fen_tokens: [B, 77] long, actions: [B] long -> bin logits [B, NUM_BINS]."""
        x = torch.cat([self.tok_emb(fen_tokens), self.act_emb(actions)[:, None]], dim=1)
        x = x + self.pos_emb.weight
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, -1])
