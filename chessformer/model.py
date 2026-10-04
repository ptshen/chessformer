"""64-square encoder and multi-task heads described in arXiv:2409.12272.

Independent PyTorch implementation. `shaw_cross_term=True` follows Eq. 6
literally; False follows the additive q/k terms in the pinned authors' code.
`deepnorm="trainer"` selects its residual/variance conventions without claiming
identical TensorFlow random draws or optimizer trajectories.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .encoding import ACTIONS


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    layers: int = 8
    heads: int = 8
    d_ff: int = 256
    position: str = "shaw"
    attention_impl: str = "dense"
    shaw_cross_term: bool = True
    attention_dim: int | None = None
    deepnorm: str = "canonical"
    value_dim: int = 32
    value_hidden: int = 128
    value_bins: int = 0

    def __post_init__(self):
        if min(self.d_model, self.layers, self.heads, self.d_ff, self.value_dim, self.value_hidden) < 1:
            raise ValueError("model dimensions must be positive")
        if self.attention_width < 1 or self.attention_width % self.heads:
            raise ValueError("attention dimension must be positive and divisible by heads")
        if self.deepnorm not in ("canonical", "trainer"):
            raise ValueError("deepnorm must be canonical or trainer")
        if self.position not in ("absolute", "relative_bias", "shaw"):
            raise ValueError("unknown position representation")
        if self.attention_impl not in ("dense", "factorized"):
            raise ValueError("unknown attention implementation")
        if self.value_bins < 0 or self.value_bins == 1:
            raise ValueError("value_bins must be 0 or >=2")

    @property
    def attention_width(self):
        return self.d_model if self.attention_dim is None else self.attention_dim


def keras_variance_scaling_(weight, scale=1.0, seed=None):
    """Keras fan_avg/truncated_normal distribution, not TensorFlow RNG parity.

    Keras treats scale as a variance multiplier. Its normal is truncated at two
    pre-truncation standard deviations, then corrected to the requested variance.
    Explicitly reseeding reproduces same-shape reuse of the trainer's seed=42.
    """
    if weight.device.type == "meta":
        return weight
    fan_out, fan_in = weight.shape
    std = math.sqrt(2 * scale / (fan_in + fan_out)) / 0.87962566103423978
    generator = None if seed is None else torch.Generator(device=weight.device).manual_seed(seed)
    return nn.init.trunc_normal_(weight, std=std, a=-2 * std, b=2 * std, generator=generator)


def relative_indices():
    squares = torch.arange(64)
    ranks, files = squares // 8, squares % 8
    return (ranks[:, None] - ranks[None, :] + 7) * 15 + files[:, None] - files[None, :] + 7


def shaw_attention(q, k, v, rq, rk, rv, indices, *, implementation="dense", cross_term=True):
    """Exact Eq. 6/7 with per-head tables [H,225,D] and QKV [B,H,64,D].

    The factorized path projects to displacement tables and scatter-sums attention
    by displacement. It avoids materializing three [H,64,64,D] tensors.
    """
    logits = q @ k.transpose(-1, -2)
    if implementation == "dense":
        aq, ak, av = rq[:, indices], rk[:, indices], rv[:, indices]
        logits = logits + torch.einsum("bhid,hijd->bhij", q, ak)
        logits = logits + torch.einsum("bhjd,hijd->bhij", k, aq)
        if cross_term:
            logits = logits + (aq * ak).sum(-1)
    elif implementation == "factorized":
        idx = indices.expand(q.shape[0], q.shape[1], -1, -1)
        logits = logits + (q @ rk.transpose(-1, -2)).gather(-1, idx)
        logits = logits + (k @ rq.transpose(-1, -2)).gather(-1, idx.transpose(-1, -2)).transpose(-1, -2)
        if cross_term:
            logits = logits + (rq * rk).sum(-1)[:, indices]
    else:
        raise ValueError(implementation)
    weights = (logits / math.sqrt(q.shape[-1])).softmax(-1)
    out = weights @ v
    if implementation == "dense":
        out = out + torch.einsum("bhij,hijd->bhid", weights, av)
    else:
        grouped = weights.new_zeros((*weights.shape[:-1], 225)).scatter_add(-1, idx, weights)
        out = out + grouped @ rv
    return out, weights


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, h, a = cfg.d_model, cfg.heads, cfg.attention_width
        self.qkv = nn.Linear(d, 3 * a, bias=False)
        self.proj = nn.Linear(a, d)
        self.register_buffer("indices", relative_indices(), persistent=False)
        if cfg.position == "shaw":
            self.rq = nn.Parameter(torch.zeros(h, 225, a // h))
            self.rk = nn.Parameter(torch.zeros(h, 225, a // h))
            self.rv = nn.Parameter(torch.zeros(h, 225, a // h))
        elif cfg.position == "relative_bias":
            self.bias = nn.Parameter(torch.zeros(h, 225))
        else:
            self.absolute = nn.Parameter(torch.zeros(64, d))

    def forward(self, x, return_attention=False):
        b, n, d = x.shape
        if self.cfg.position == "absolute":
            x = x + self.absolute
        a = self.cfg.attention_width
        q, k, v = self.qkv(x).view(b, n, 3, self.cfg.heads, a // self.cfg.heads).permute(2, 0, 3, 1, 4)
        if self.cfg.position == "shaw":
            # Autocast linear outputs and learned tables must share a dtype.
            out, weights = shaw_attention(q, k, v, self.rq.to(q.dtype), self.rk.to(q.dtype),
                                         self.rv.to(q.dtype), self.indices,
                                         implementation=self.cfg.attention_impl,
                                         cross_term=self.cfg.shaw_cross_term)
        else:
            bias = self.bias[:, self.indices].to(q.dtype) if self.cfg.position == "relative_bias" else None
            if return_attention:
                logits = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
                weights = (logits if bias is None else logits + bias).softmax(-1)
                out = weights @ v
            else:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
                weights = None
        return self.proj(out.transpose(1, 2).reshape(b, n, a)), weights if return_attention else None


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = Attention(cfg)
        self.norm1 = nn.RMSNorm(cfg.d_model, eps=1e-5)
        self.norm2 = nn.RMSNorm(cfg.d_model, eps=1e-5)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model)
        self.alpha = (2 * cfg.layers) ** 0.25
        self.trainer_scaling = cfg.deepnorm == "trainer"

    def forward(self, x, return_attention=False):
        a, weights = self.attn(x, return_attention)
        if self.trainer_scaling:
            x = self.norm1(x + a / self.alpha)
            return self.norm2(x + self.down(F.mish(self.up(x))) / self.alpha), weights
        x = self.norm1(self.alpha * x + a)
        return self.norm2(self.alpha * x + self.down(F.mish(self.up(x)))), weights


class PolicyHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.promotion = nn.Linear(d, 4, bias=False)
        self.register_buffer("src", torch.tensor([a[0] for a in ACTIONS]), persistent=False)
        self.register_buffer("dst", torch.tensor([a[1] for a in ACTIONS]), persistent=False)
        self.register_buffer("promotion_dst", torch.tensor([a[1] for a in ACTIONS[1792:]]), persistent=False)
        self.register_buffer("promotion_kind", torch.tensor([0, 1, 2] * 22), persistent=False)

    def forward(self, x):
        q, k = self.query(x), self.key(x)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        selected = logits[:, self.src, self.dst]
        offsets = self.promotion(k)
        extra = offsets[:, self.promotion_dst, self.promotion_kind] + offsets[:, self.promotion_dst, 3]
        return torch.cat((selected[:, :1792], selected[:, 1792:] + extra), -1)


class ValueHead(nn.Module):
    def __init__(self, cfg, result=False):
        super().__init__()
        self.embed = nn.Linear(cfg.d_model, cfg.value_dim)
        self.hidden = nn.Linear(64 * cfg.value_dim, cfg.value_hidden)
        self.output = nn.Linear(cfg.value_hidden, 3 if result else 1)
        self.error = None if result else nn.Linear(cfg.value_hidden, 1)
        self.category = nn.Linear(cfg.value_hidden, cfg.value_bins) if cfg.value_bins and not result else None

    def forward(self, x):
        x = F.mish(self.hidden(F.mish(self.embed(x)).flatten(1)))
        if self.error is None:
            return {"logits": self.output(x)}
        out = {"value": self.output(x).squeeze(-1).tanh(),
               "error": self.error(x).squeeze(-1).sigmoid()}
        if self.category is not None:
            out["category"] = self.category(x)
        return out


class Chessformer(nn.Module):
    def __init__(self, cfg=ModelConfig()):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Linear(112, cfg.d_model)
        self.input_scale = nn.Parameter(torch.ones(64, cfg.d_model))
        self.input_offset = nn.Parameter(torch.zeros(64, cfg.d_model))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.policy_embedding = nn.Linear(cfg.d_model, cfg.d_model)
        self.policy = PolicyHead(cfg.d_model)
        self.soft_policy = PolicyHead(cfg.d_model)
        self.result = ValueHead(cfg, result=True)
        self.q = ValueHead(cfg)
        self.st = ValueHead(cfg)
        self.reset_parameters()

    def reset_parameters(self):
        trainer = self.cfg.deepnorm == "trainer"
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if trainer:
                    keras_variance_scaling_(module.weight)
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        beta = (8 * self.cfg.layers) ** -0.25
        for block in self.blocks:
            # Separate Q/K Xavier initialization; only V, O and FFN get DeepNorm gain.
            for part in block.attn.qkv.weight.chunk(3):
                if trainer:
                    keras_variance_scaling_(part)
                else:
                    nn.init.xavier_normal_(part)
            if trainer:
                keras_variance_scaling_(block.attn.qkv.weight[2 * self.cfg.attention_width:], beta, seed=42)
            else:
                with torch.no_grad():
                    block.attn.qkv.weight[2 * self.cfg.attention_width:].mul_(beta)
            for layer in (block.attn.proj, block.up, block.down):
                if trainer:
                    keras_variance_scaling_(layer.weight, beta, seed=42)
                else:
                    nn.init.xavier_normal_(layer.weight, gain=beta)

    def forward(self, inputs, return_attention=False, *, heads=None):
        """Compute all training heads by default, or explicitly selected outputs.

        Selecting policy or WDL at inference skips unused head computations while
        preserving the shared encoder, parameter layout, and requested logits.
        """
        available = {"policy", "soft_policy", "wdl", "q", "st"}
        selected = available if heads is None else frozenset(heads)
        if not selected or not selected <= available:
            raise ValueError("heads must be a nonempty selection of policy, soft_policy, wdl, q, st")
        if inputs.ndim != 3 or inputs.shape[1:] != (64, 112):
            raise ValueError("expected inputs [batch,64,112]")
        x = F.mish(self.embedding(inputs)) * self.input_scale + self.input_offset
        maps = []
        for block in self.blocks:
            x, weights = block(x, return_attention)
            if return_attention:
                maps.append(weights)
        out = {}
        if selected & {"policy", "soft_policy"}:
            p = F.mish(self.policy_embedding(x))
            if "policy" in selected:
                out["policy"] = self.policy(p)
            if "soft_policy" in selected:
                out["soft_policy"] = self.soft_policy(p)
        if "wdl" in selected:
            out["wdl"] = self.result(x)["logits"]
        if "q" in selected:
            out["q"] = self.q(x)
        if "st" in selected:
            out["st"] = self.st(x)
        if return_attention:
            out["attention"] = torch.stack(maps, 1)
        return out

    def constrain_parameters(self):
        """Match the nonnegative input/RMS gain constraint used upstream."""
        with torch.no_grad():
            self.input_scale.clamp_(min=0)
            for module in self.modules():
                if isinstance(module, nn.RMSNorm):
                    module.weight.clamp_(min=0)
