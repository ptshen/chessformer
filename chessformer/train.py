"""Deterministic local/GPU trainer with accumulation, SWA, and restart state."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import time

import torch
from torch.optim.swa_utils import AveragedModel

from .data import load_dataset, assert_disjoint, source_fingerprint
from .losses import LossConfig, multitask_loss
from .model import Chessformer, ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 200
    batch_size: int = 32
    microbatch_size: int = 16
    lr: float = 5e-4
    lr_milestones: tuple = (160, 180)
    lr_values: tuple = (1.58e-4, 5e-5)
    swa_start: int = 160
    swa_every: int = 10
    checkpoint_every: int = 100
    eval_every: int = 100
    seed: int = 0
    precision: str = "float32"

    def __post_init__(self):
        object.__setattr__(self, "lr_milestones", tuple(self.lr_milestones))
        object.__setattr__(self, "lr_values", tuple(self.lr_values))
        if min(self.steps, self.batch_size, self.microbatch_size, self.swa_every,
               self.checkpoint_every, self.eval_every) < 1:
            raise ValueError("training sizes and intervals must be positive")
        if len(self.lr_milestones) != len(self.lr_values) or tuple(sorted(self.lr_milestones)) != self.lr_milestones:
            raise ValueError("invalid LR schedule")
        if self.precision not in ("float32", "bfloat16"):
            raise ValueError("precision must be float32 or bfloat16")


def lr_at(step, cfg):
    lr = cfg.lr
    for milestone, value in zip(cfg.lr_milestones, cfg.lr_values):
        if step >= milestone:
            lr = value
    return lr


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def autocast(device, precision):
    return torch.autocast(torch.device(device).type, dtype=torch.bfloat16) if precision == "bfloat16" else nullcontext()


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=32, precision="float32"):
    was_training = model.training
    model.eval()
    sums = {"policy_ce": 0., "policy_kl": 0., "policy_accuracy": 0., "wdl_ce": 0., "wdl_accuracy": 0.}
    q_counts = {"q": 0, "st": 0}
    for name in q_counts:
        sums[name + "_mse"] = 0.
    for start in range(0, len(dataset), batch_size):
        batch = dataset.batch(range(start, min(len(dataset), start + batch_size)), device)
        with autocast(device, precision):
            output = model(batch["inputs"])
        _, metrics = multitask_loss(output, batch)
        size = len(batch["inputs"])
        for key in ("policy_ce", "policy_kl", "wdl_ce"):
            sums[key] += float(metrics[key]) * size
        selected = output["policy"].masked_fill(~batch["legal"], -1e9).argmax(-1)
        sums["policy_accuracy"] += int((selected == batch["policy"].argmax(-1)).sum())
        sums["wdl_accuracy"] += int((output["wdl"].argmax(-1) == batch["wdl"].argmax(-1)).sum())
        for name in q_counts:
            count = int(batch[name + "_valid"].sum())
            sums[name + "_mse"] += float(metrics[name + "_mse"]) * count
            q_counts[name] += count
    model.train(was_training)
    result = {k: v / len(dataset) for k, v in sums.items() if not k.endswith("_mse")}
    for name, count in q_counts.items():
        result[name + "_mse"] = sums[name + "_mse"] / count if count else None
        result[name + "_count"] = count
    return result


def run_training(model_cfg, train_cfg, loss_cfg, train, validation, out, device="cpu", resume=False, stop_after=None):
    """stop_after interrupts at an optimizer boundary; config/schedule stay fixed."""
    cfg = train_cfg
    assert_disjoint(train, validation)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoint.pt"
    if not resume and (checkpoint.exists() or (out / "metrics.jsonl").exists()):
        raise FileExistsError("run exists; use --resume or choose a fresh --out")
    if resume and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    torch.manual_seed(cfg.seed)
    model = Chessformer(model_cfg).to(device)
    opt = torch.optim.NAdam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-7, weight_decay=0)
    swa = AveragedModel(model)
    generator = torch.Generator().manual_seed(cfg.seed + 1)
    spec = {"model": asdict(model_cfg), "training": asdict(cfg), "loss": asdict(loss_cfg)}
    provenance = {"encoding": train.encoding, "train_sha256": train.fingerprint,
                  "validation_sha256": validation.fingerprint, "source_sha256": source_fingerprint()}
    manifest = {**spec, **provenance, "train_positions": len(train), "validation_positions": len(validation),
                "train_games": train.game_count, "validation_games": validation.game_count,
                "parameters": sum(p.numel() for p in model.parameters()), "device": str(device),
                "torch": torch.__version__, "platform": platform.platform(), "threads": torch.get_num_threads(),
                "sources": train.sources, "available_aux": train.available_aux}
    start_step = 0
    elapsed_before = 0.
    if resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["spec"] != spec or state["provenance"] != provenance:
            raise ValueError("resume config or dataset fingerprint changed")
        if state["device"] != str(device):
            raise ValueError("exact resume requires the original device")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        swa.load_state_dict(state["swa"])
        generator.set_state(state["sampler_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.device(device).type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_step = state["step"]
        elapsed_before = state["elapsed_seconds"]
    else:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logpath = out / "metrics.jsonl"
    # An interrupted process can leave metrics after its last durable checkpoint.
    if resume and logpath.exists():
        entries = [json.loads(line) for line in logpath.read_text().splitlines()]
        logpath.write_text("".join(json.dumps(e) + "\n" for e in entries if e["step"] <= start_step))

    def log(entry):
        with logpath.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    if not resume:
        log({"step": 0, "validation": evaluate(model, validation, device, cfg.microbatch_size, cfg.precision)})
    end_step = min(cfg.steps, stop_after) if stop_after is not None else cfg.steps
    if end_step < start_step:
        raise ValueError("stop_after precedes the restored step")
    started = time.perf_counter()
    latest = None
    for step in range(start_step, end_step):
        model.train()
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)
        # Stateless-sized sampling, with explicit generator state persisted on resume.
        indices = torch.randint(len(train), (cfg.batch_size,), generator=generator)
        valid_counts = train.aux_counts(indices)
        # Each microbatch contributes its fraction of the effective batch. Missing
        # aux losses need their own denominators, handled by scaling coefficients.
        opt.zero_grad(set_to_none=True)
        train_loss = 0.
        for i in range(0, cfg.batch_size, cfg.microbatch_size):
            batch = train.batch(indices[i:i + cfg.microbatch_size], device)
            with autocast(device, cfg.precision):
                output = model(batch["inputs"])
            loss, metrics = multitask_loss(output, batch, loss_cfg)
            fraction = len(batch["inputs"]) / cfg.batch_size
            loss = loss * fraction
            for name, n_valid in valid_counts.items():
                adjustment = float(batch[name + "_valid"].sum()) / max(n_valid, 1) - fraction
                loss = loss + adjustment * (loss_cfg.value * metrics[name + "_mse"] + loss_cfg.error * metrics[name + "_error_mse"])
                if name + "_category_ce" in metrics:
                    loss = loss + adjustment * loss_cfg.category * metrics[name + "_category_ce"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step + 1}")
            loss.backward()
            train_loss += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        opt.step()
        model.constrain_parameters()
        completed = step + 1
        if completed >= cfg.swa_start and (completed - cfg.swa_start) % cfg.swa_every == 0:
            swa.update_parameters(model)
        if completed % cfg.eval_every == 0 or completed == end_step:
            latest = evaluate(model, validation, device, cfg.microbatch_size, cfg.precision)
            entry = {"step": completed, "train_loss": train_loss, "grad_norm": float(norm),
                     "lr": opt.param_groups[0]["lr"], "validation": latest,
                     "elapsed_seconds": elapsed_before + time.perf_counter() - started}
            if int(swa.n_averaged):
                entry["swa_validation"] = evaluate(swa.module, validation, device, cfg.microbatch_size, cfg.precision)
            log(entry)
        if completed % cfg.checkpoint_every == 0 or completed == end_step:
            atomic_save({"spec": spec, "provenance": provenance, "device": str(device), "step": completed,
                         "model": model.state_dict(), "optimizer": opt.state_dict(), "swa": swa.state_dict(),
                         "sampler_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                         "cuda_rng": torch.cuda.get_rng_state_all() if torch.device(device).type == "cuda" else [],
                         "elapsed_seconds": elapsed_before + time.perf_counter() - started}, checkpoint)
    if end_step == cfg.steps:
        chosen = swa.module if int(swa.n_averaged) else model
        atomic_save({"model": chosen.state_dict(), "model_config": asdict(model_cfg), "encoding": train.encoding,
                     "step": end_step, "swa_samples": int(swa.n_averaged), "provenance": provenance}, out / "model.pt")
    return {"step": end_step, "validation": latest, "parameters": manifest["parameters"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--position", choices=["absolute", "relative_bias", "shaw"])
    parser.add_argument("--attention-impl", choices=["dense", "factorized"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--allow-missing-aux", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    spec = json.loads(Path(args.config).read_text())
    if args.position:
        spec["model"]["position"] = args.position
    if args.attention_impl:
        spec["model"]["attention_impl"] = args.attention_impl
    if args.seed is not None:
        spec["training"]["seed"] = args.seed
    train = load_dataset(args.train, "train", require_aux=not args.allow_missing_aux)
    validation = load_dataset(args.validation, "validation", require_aux=not args.allow_missing_aux)
    run_training(ModelConfig(**spec["model"]), TrainConfig(**spec["training"]), LossConfig(**spec.get("loss", {})),
                 train, validation, args.out, args.device, args.resume, args.stop_after)


if __name__ == "__main__":
    main()
