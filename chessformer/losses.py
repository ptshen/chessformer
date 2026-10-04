"""Appendix B multi-task objective with explicit missing-target masks."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossConfig:
    policy: float = 1.0
    soft_policy: float = 8.0
    temperature: float = 4.0
    wdl: float = 1.0
    value: float = 1.0
    error: float = 1.0
    category: float = 0.0


def policy_loss(logits, target, legal, temperature=1.0):
    # Zero-probability legal moves remain in the softmax denominator.
    logits = logits.float().masked_fill(~legal, -1e9)
    target = target.float().pow(1 / temperature) * legal
    target = target / target.sum(-1, keepdim=True)
    logp = logits.log_softmax(-1)
    ce = -(target * logp).sum(-1).mean()
    entropy = -(target * target.clamp_min(1e-30).log()).sum(-1).mean()
    return ce, ce - entropy


def multitask_loss(output, batch, cfg=LossConfig()):
    ce, kl = policy_loss(output["policy"], batch["policy"], batch["legal"])
    soft_ce, soft_kl = policy_loss(output["soft_policy"], batch["policy"], batch["legal"], cfg.temperature)
    wdl = -(batch["wdl"].float() * output["wdl"].float().log_softmax(-1)).sum(-1).mean()
    total = cfg.policy * ce + cfg.soft_policy * soft_ce + cfg.wdl * wdl
    metrics = {"policy_ce": ce, "policy_kl": kl, "soft_policy_kl": soft_kl, "wdl_ce": wdl}
    for name in ("q", "st"):
        mask = batch[name + "_valid"].float()
        denom = mask.sum().clamp_min(1)
        predicted = output[name]["value"].float()
        target = batch[name].float()
        squared = (predicted - target).square()
        mse = (squared * mask).sum() / denom
        error = ((output[name]["error"].float() - squared.detach()).square() * mask).sum() / denom
        total = total + cfg.value * mse + cfg.error * error
        metrics[name + "_mse"] = mse
        metrics[name + "_error_mse"] = error
        if "category" in output[name]:
            logits = output[name]["category"].float()
            buckets = ((target + 1) * (logits.shape[-1] / 2)).long().clamp(0, logits.shape[-1] - 1)
            cat = (F.cross_entropy(logits, buckets, reduction="none") * mask).sum() / denom
            total = total + cfg.category * cat
            metrics[name + "_category_ce"] = cat
    metrics["loss"] = total
    return total, metrics
