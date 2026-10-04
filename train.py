"""Pack labeled shards into flat training arrays and train on one A100 with bf16.

Usage:
    modal run train.py --stage pack
    modal run train.py --stage train
    modal run train.py --stage bench   # fp32 vs bf16 step-time comparison
"""

import math
import os
import time

import modal

BATCH = 4096
EPOCHS = 10
LR = 3e-4
WARMUP = 1000
EVAL_BOARDS = 50_000
CKPT_EVERY = 10_000

app = modal.App("chessformer-train")
vol = modal.Volume.from_name("cclr-data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.1", "numpy", "scipy")
    .add_local_python_source("common", "model")
)


def _load_split(split: str, num_shards: int = 64):
    import numpy as np

    toks, counts, moves, wins = [], [], [], []
    for i in range(num_shards):
        d = np.load(f"/data/labels/{split}_{i:02d}.npz")
        n_boards = int(d["done"])
        # counts is uint16; cumsum/sum must be int64 or index math silently promotes
        c = d["counts"][:n_boards].astype(np.int64)
        n_samples = int(c.sum())
        toks.append(d["tokens"][:n_boards])
        counts.append(c)
        moves.append(d["move_ids"][:n_samples])
        wins.append(d["wins"][:n_samples])
    return [np.concatenate(x) for x in (toks, counts, moves, wins)]


@app.function(image=image, volumes={"/data": vol}, timeout=3600, memory=65536)
def pack(split: str = "both") -> None:
    import numpy as np

    if split in ("train", "both"):
        toks, counts, moves, wins = _load_split("train")
        board_idx = np.repeat(np.arange(len(toks), dtype=np.uint32), counts)
        assert len(board_idx) == len(moves) == len(wins)
        np.savez(
            "/data/packed_train.npz",
            board_tokens=toks, s_board=board_idx, s_move=moves, s_win=wins,
        )
        print(f"train: {len(toks)} boards, {len(moves)} action-values")

    if split in ("test", "both"):
        toks, counts, moves, wins = _load_split("test")
        toks, counts = toks[:EVAL_BOARDS], counts[:EVAL_BOARDS]
        n = int(counts.sum())
        np.savez(
            "/data/packed_eval.npz",
            board_tokens=toks, counts=counts, move_ids=moves[:n], wins=wins[:n],
        )
        print(f"eval: {len(toks)} boards, {n} action-values")
    vol.commit()


def hl_gauss(win, edges, sigma):
    import torch

    z = (edges[None, :] - win[:, None]) / (sigma * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + torch.erf(z))
    t = cdf[:, 1:] - cdf[:, :-1]
    return t / t.sum(-1, keepdim=True)


@app.function(image=image, volumes={"/data": vol}, timeout=20 * 3600, gpu="A100", memory=65536)
def train() -> None:
    import numpy as np
    import torch

    import common
    from model import ActionValueTransformer

    dev = "cuda"
    torch.manual_seed(0)
    d = np.load("/data/packed_train.npz")
    board_tokens = torch.from_numpy(d["board_tokens"]).to(dev)
    s_board = torch.from_numpy(d["s_board"].astype(np.int64)).to(dev)
    s_move = torch.from_numpy(d["s_move"].astype(np.int64)).to(dev)
    s_win = torch.from_numpy(d["s_win"].astype(np.float32)).to(dev)
    n = len(s_move)
    steps_per_epoch = n // BATCH
    total_steps = EPOCHS * steps_per_epoch
    print(f"{n} samples, {steps_per_epoch} steps/epoch, {total_steps} total")

    ev = np.load("/data/packed_eval.npz")
    ev_sub = 2000  # quick in-training eval on a slice of the eval set
    ev_counts = ev["counts"][:ev_sub].astype(np.int64)
    ev_off = np.concatenate([[0], np.cumsum(ev_counts)])
    ev_tokens = torch.from_numpy(ev["board_tokens"][:ev_sub]).long().to(dev)
    ev_moves = torch.from_numpy(ev["move_ids"][: ev_off[-1]].astype(np.int64)).to(dev)
    ev_wins = ev["wins"][: ev_off[-1]].astype(np.float32)
    ev_board_of = torch.from_numpy(np.repeat(np.arange(ev_sub), ev_counts)).to(dev)

    model = ActionValueTransformer().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, s / WARMUP)
        * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))),
    )
    start_step = 0
    if os.path.exists("/data/ckpt/model.pt"):
        ck = torch.load("/data/ckpt/model.pt", map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_step = ck["step"]
        print(f"resumed from step {start_step}")
    cmodel = torch.compile(model)

    edges = torch.linspace(0.0, 1.0, common.NUM_BINS + 1, device=dev)
    centers = torch.from_numpy(common.BIN_CENTERS).float().to(dev)

    @torch.no_grad()
    def quick_eval():
        model.eval()
        preds = []
        for i in range(0, len(ev_moves), 8192):
            logits = cmodel(ev_tokens[ev_board_of[i : i + 8192]], ev_moves[i : i + 8192])
            preds.append((logits.float().softmax(-1) @ centers).cpu().numpy())
        preds = np.concatenate(preds)
        hits = sum(
            preds[ev_off[b] : ev_off[b + 1]].argmax() == ev_wins[ev_off[b] : ev_off[b + 1]].argmax()
            for b in range(ev_sub)
        )
        model.train()
        return hits / ev_sub

    os.makedirs("/data/ckpt", exist_ok=True)
    g = torch.Generator(device=dev)
    g.manual_seed(1)
    perm = torch.randperm(n, device=dev, generator=g)
    t0, losses = time.time(), []
    for step in range(start_step, total_steps):
        i = step % steps_per_epoch
        if i == 0 and step > start_step:
            perm = torch.randperm(n, device=dev, generator=g)
        idx = perm[i * BATCH : (i + 1) * BATCH]
        fen = board_tokens[s_board[idx]].long()
        targets = hl_gauss(s_win[idx], edges, common.HL_GAUSS_SIGMA)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = cmodel(fen, s_move[idx])
        loss = -(targets * logits.float().log_softmax(-1)).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (step + 1) % 500 == 0:
            dt = (time.time() - t0) / len(losses)
            print(
                f"step {step + 1}/{total_steps} loss {sum(losses) / len(losses):.4f} "
                f"{dt * 1000:.0f} ms/step lr {sched.get_last_lr()[0]:.2e}", flush=True,
            )
            t0, losses = time.time(), []
        if (step + 1) % 20_000 == 0:
            print(f"step {step + 1}: quick action accuracy {quick_eval():.4f}", flush=True)
        if (step + 1) % CKPT_EVERY == 0 or step + 1 == total_steps:
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "step": step + 1},
                "/data/ckpt/model.pt",
            )
            vol.commit()
    print(f"final quick action accuracy: {quick_eval():.4f}")


@app.function(image=image, volumes={"/data": vol}, timeout=3600, gpu="A100")
def evaluate() -> None:
    """Top-move match vs the Stockfish oracle on the full held-out eval set."""
    import numpy as np
    import torch

    import common
    from model import ActionValueTransformer

    dev = "cuda"
    ev = np.load("/data/packed_eval.npz")
    counts = ev["counts"].astype(np.int64)
    off = np.concatenate([[0], np.cumsum(counts)])
    n_boards, n_samples = len(counts), int(off[-1])
    tokens = torch.from_numpy(ev["board_tokens"]).long().to(dev)
    moves = torch.from_numpy(ev["move_ids"].astype(np.int64)).to(dev)
    wins = ev["wins"].astype(np.float32)
    board_of = torch.from_numpy(np.repeat(np.arange(n_boards), counts)).to(dev)

    model = ActionValueTransformer().to(dev)
    ck = torch.load("/data/ckpt/model.pt", map_location=dev)
    model.load_state_dict(ck["model"])
    model.eval()
    cmodel = torch.compile(model)
    centers = torch.from_numpy(common.BIN_CENTERS).float().to(dev)

    preds = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, n_samples, 16384):
            logits = cmodel(tokens[board_of[i : i + 16384]], moves[i : i + 16384])
            preds.append((logits.float().softmax(-1) @ centers).cpu().numpy())
    preds = np.concatenate(preds)
    hits = sum(
        preds[off[b] : off[b + 1]].argmax() == wins[off[b] : off[b + 1]].argmax()
        for b in range(n_boards)
    )
    print(f"step {ck['step']}: {hits}/{n_boards} top-move match = {hits / n_boards:.4f}")


@app.function(image=image, volumes={"/data": vol}, timeout=1800, gpu="A100")
def bench() -> None:
    import torch

    import common
    from model import ActionValueTransformer

    dev = "cuda"
    model = ActionValueTransformer().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    cmodel = torch.compile(model)
    edges = torch.linspace(0.0, 1.0, common.NUM_BINS + 1, device=dev)
    fen = torch.randint(0, common.VOCAB_SIZE, (BATCH, 77), device=dev)
    act = torch.randint(0, common.NUM_ACTIONS, (BATCH,), device=dev)
    win = torch.rand(BATCH, device=dev)

    def run(use_bf16, steps=60):
        for s in range(steps):
            if s == 10:
                torch.cuda.synchronize()
                t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = cmodel(fen, act)
                loss = -(hl_gauss(win, edges, common.HL_GAUSS_SIGMA)
                         * logits.float().log_softmax(-1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        return (time.time() - t0) / (steps - 10)

    fp32 = run(False)
    bf16 = run(True)
    print(f"fp32 {fp32 * 1000:.1f} ms/step, bf16 {bf16 * 1000:.1f} ms/step "
          f"({100 * (1 - bf16 / fp32):.0f}% faster)")


@app.local_entrypoint()
def main(stage: str, split: str = "both"):
    if stage == "pack":
        pack.remote(split)
    else:
        {"train": train, "eval": evaluate, "bench": bench}[stage].remote()
