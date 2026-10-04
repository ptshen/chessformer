"""Measure selected-head inference with unchanged weights and paired timings.

Arithmetic counts use PyTorch's registered FLOP rules (multiply/add count as two).
For Shaw these cover matrix products, not elementwise operations, nonlinearities,
normalization, gathers/scatters, encoding, or memory traffic. They are not a full
instruction count, and the paper's FLOPS convention is not assumed equivalent.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import time

import chess
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from .data import sha256_file
from .model import Chessformer, ModelConfig
from .model_reference import exercise_relative_parameters, flattened
from .play import choose_move, encoding_functions, load_model


HEADS = {"all": None, "policy": ("policy",), "wdl": ("wdl",)}


def weight_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def arithmetic_count(model, inputs, heads=None):
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        model(inputs, heads=heads)
    return {"registered_flops": counter.get_total_flops(),
            "operators": {str(key): count for key, count in counter.get_flop_counts()["Global"].items()}}


def measure(callbacks, *, iterations, warmup, device, seed=73):
    rng = np.random.default_rng(seed)
    times = {name: [] for name in callbacks}

    def sync():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        elif torch.device(device).type == "mps":
            torch.mps.synchronize()

    with torch.inference_mode():
        for repeat in range(warmup + iterations):
            # Every round includes every condition, with randomized order.
            for name in rng.permutation(list(callbacks)):
                sync()
                start = time.perf_counter()
                callbacks[name]()
                sync()
                if repeat >= warmup:
                    times[name].append((time.perf_counter() - start) * 1000)
    return times


def summarize(times, baseline="all"):
    base = np.asarray(times[baseline])
    rows = {}
    rng = np.random.default_rng(97)
    indices = rng.integers(0, len(base), size=(2000, len(base)))
    for name, samples in times.items():
        samples = np.asarray(samples)
        ratios = np.median(base[indices], axis=1) / np.median(samples[indices], axis=1)
        rows[name] = {"median_ms": float(np.median(samples)), "p10_ms": float(np.quantile(samples, 0.1)),
                      "p90_ms": float(np.quantile(samples, 0.9)), "samples_ms": samples.tolist(),
                      "speedup_vs_all": float(np.median(base) / np.median(samples)),
                      "speedup_paired_bootstrap_95": np.quantile(ratios, [0.025, 0.975]).tolist()}
    return rows


def benchmark(model, *, iterations=30, warmup=5, threads=4):
    if iterations < 5 or warmup < 1 or threads < 1:
        raise ValueError("benchmark needs >=5 rounds, >=1 warmup, and positive threads")
    torch.set_num_threads(threads)
    device = next(model.parameters()).device
    encode, _ = encoding_functions(model.input_encoding)
    opening = chess.Board()
    for move in "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7".split():
        opening.push_uci(move)
    start = chess.Board()
    children = []
    for move in start.legal_moves:
        child = start.copy(stack=True)
        child.push(move)
        children.append(child)
    tensors = {"single_position": torch.from_numpy(np.stack([encode(opening)])).to(device),
               "starting_position_children": torch.from_numpy(np.stack([encode(b) for b in children])).to(device)}
    implementations = ("dense", "factorized") if model.cfg.position == "shaw" else (model.cfg.attention_impl,)
    rows, dense_outputs = [], {}
    for implementation in implementations:
        candidate = Chessformer(replace(model.cfg, attention_impl=implementation)).to(device).eval()
        candidate.load_state_dict(model.state_dict())
        candidate.input_encoding = model.input_encoding
        for name, inputs in tensors.items():
            with torch.inference_mode():
                all_outputs = flattened(candidate(inputs))
                head_error = 0.
                for label, heads in HEADS.items():
                    for key, value in flattened(candidate(inputs, heads=heads)).items():
                        head_error = max(head_error, float((value - all_outputs[key]).abs().max()))
                        torch.testing.assert_close(value, all_outputs[key], atol=0, rtol=0)
                implementation_error = 0.
                if name not in dense_outputs:
                    dense_outputs[name] = all_outputs
                else:
                    for key, value in all_outputs.items():
                        implementation_error = max(implementation_error, float((value - dense_outputs[name][key]).abs().max()))
                        torch.testing.assert_close(value, dense_outputs[name][key], atol=1e-5, rtol=1e-4)
            callbacks = {label: lambda heads=heads: candidate(inputs, heads=heads) for label, heads in HEADS.items()}
            times = summarize(measure(callbacks, iterations=iterations, warmup=warmup, device=device))
            for label, heads in HEADS.items():
                times[label]["arithmetic"] = arithmetic_count(candidate, inputs, heads)
            rows.append({"input": name, "batch": len(inputs), "implementation": implementation,
                         "selected_head_max_abs_difference": head_error,
                         "max_abs_difference_vs_dense": implementation_error, "timings": times})
            print(json.dumps({"completed_forward_case": name, "implementation": implementation}), flush=True)
    moves = []
    for strategy in ("policy", "value"):
        expected = choose_move(model, opening, strategy, select_heads=False)
        actual = choose_move(model, opening, strategy)
        if expected != actual:
            raise AssertionError("selected heads changed the chosen move")
        callbacks = {"all": lambda: choose_move(model, opening, strategy, select_heads=False),
                     "selected": lambda: choose_move(model, opening, strategy)}
        times = summarize(measure(callbacks, iterations=iterations, warmup=warmup, device=device))
        moves.append({"strategy": strategy, "implementation": model.cfg.attention_impl,
                      "move": actual.uci(), "legal_moves": opening.legal_moves.count(), "timings": times})
    return {"scope": "CPU/GPU inference implementation benchmark; no playing-strength measurement",
            "torch": torch.__version__, "platform": platform.platform(), "device": str(device),
            "threads": threads, "dtype": str(next(model.parameters()).dtype), "iterations": iterations, "warmup": warmup,
            "timing_protocol": "interleaved conditions in seeded random order; bootstrap resamples paired rounds within this process",
            "config": asdict(model.cfg), "encoding": model.input_encoding,
            "parameters_stored": sum(p.numel() for p in model.parameters()), "weights_sha256": weight_digest(model),
            "input_sha256": {name: hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for name, value in tensors.items()},
            "opening_fen": opening.fen(), "opening_history": [m.uci() for m in opening.move_stack],
            "arithmetic_convention": "registered PyTorch FLOPs; multiply+add=2; matmul-only for Shaw; omitted elementwise/nonlinear/norm/gather/scatter/encoding/memory cost",
            "paper_flops_convention_verified": False, "forward_results": rows, "move_selection_results": moves,
            "source_sha256": {str(p): sha256_file(p) for p in map(Path, ["chessformer/model.py", "chessformer/play.py",
                                                                        "chessformer/inference_benchmark.py"])}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("benchmark output exists; choose a new output path")
    torch.set_num_threads(args.threads)
    torch.manual_seed(103)
    if args.checkpoint:
        model = load_model(args.checkpoint, args.device)
        provenance = {"checkpoint": args.checkpoint, "checkpoint_sha256": sha256_file(args.checkpoint)}
    else:
        model = Chessformer(ModelConfig(**json.loads(Path(args.config).read_text())["model"]))
        exercise_relative_parameters(model, seed=103)
        model = model.to(args.device).eval()
        model.input_encoding = "lc0-training-v1:4"
        provenance = {"config_path": args.config, "config_sha256": sha256_file(args.config),
                      "weights": "random initialization, nonzero relative tables; not trained", "seed": 103}
    result = benchmark(model, iterations=args.iterations, threads=args.threads)
    result["provenance"] = provenance
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"report": str(out), "parameters": result["parameters_stored"]}))
