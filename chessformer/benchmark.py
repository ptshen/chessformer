"""Benchmark dense versus displacement-table Shaw attention (same mathematics)."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

import torch

from .model import relative_indices, shaw_attention


def benchmark(batch_sizes=(1, 16, 32), heads=8, depth=32, iterations=20, device="cpu", threads=4):
    torch.set_num_threads(threads)
    torch.manual_seed(7)
    idx = relative_indices().to(device)
    rows = []

    def sync():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()

    for batch in batch_sizes:
        values = [torch.randn(batch, heads, 64, depth, device=device, requires_grad=True) for _ in range(3)]
        values += [torch.randn(heads, 225, depth, device=device, requires_grad=True) for _ in range(3)]
        dense = shaw_attention(*values, idx, implementation="dense")[0]
        factorized = shaw_attention(*values, idx, implementation="factorized")[0]
        max_error = float((dense - factorized).abs().max().detach())
        torch.testing.assert_close(dense, factorized, atol=1e-5, rtol=1e-4)
        del dense, factorized
        for impl in ("dense", "factorized"):
            for backward in (False, True):
                times = []
                for repeat in range(iterations + 3):
                    for tensor in values:
                        tensor.grad = None
                    sync()
                    started = time.perf_counter()
                    with torch.set_grad_enabled(backward):
                        out = shaw_attention(*values, idx, implementation=impl)[0]
                        if backward:
                            out.square().mean().backward()
                    sync()
                    if repeat >= 3:
                        times.append((time.perf_counter() - started) * 1000)
                rows.append({"batch": batch, "implementation": impl, "backward": backward,
                             "median_ms": statistics.median(times), "min_ms": min(times),
                             "max_abs_output_difference": max_error})
    return {"torch": torch.__version__, "platform": platform.platform(), "device": str(device),
            "threads": threads, "heads": heads, "depth": depth, "dtype": "float32",
            "iterations": iterations, "warmup": 3, "results": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/attention-benchmark.json")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    result = benchmark(iterations=args.iterations, device=args.device, threads=args.threads)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
