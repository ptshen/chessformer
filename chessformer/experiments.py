"""Run matched-seed local position ablations and retain per-run evidence."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics

import torch

from .data import load_dataset, sha256_file
from .losses import LossConfig
from .model import ModelConfig
from .train import TrainConfig, run_training


def ablate(config, train_path, validation_path, out, seeds=(0, 1, 2), allow_missing_aux=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(config).read_text())
    train = load_dataset(train_path, "train", require_aux=not allow_missing_aux)
    validation = load_dataset(validation_path, "validation", require_aux=not allow_missing_aux)
    rows = []
    for seed in seeds:
        for position in ("absolute", "relative_bias", "shaw"):
            run_dir = out / f"{position}-seed{seed}"
            mc = replace(ModelConfig(**spec["model"]), position=position)
            tc = replace(TrainConfig(**spec["training"]), seed=seed)
            run_training(mc, tc, LossConfig(**spec.get("loss", {})), train, validation, run_dir)
            manifest = json.loads((run_dir / "manifest.json").read_text())
            metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
            last = metrics[-1]
            rows.append({"position": position, "seed": seed, "parameters": manifest["parameters"],
                         "source_sha256": manifest["source_sha256"],
                         "step": last["step"], "initial": metrics[0]["validation"],
                         "final": last["validation"], "swa": last.get("swa_validation"),
                         "training_seconds": last["elapsed_seconds"],
                         "checkpoint_sha256": sha256_file(run_dir / "model.pt")})
    means = []
    for position in ("absolute", "relative_bias", "shaw"):
        selected = [row for row in rows if row["position"] == position]
        summary = {"position": position, "parameters": selected[0]["parameters"], "seeds": len(selected)}
        for key in ("policy_accuracy", "policy_kl", "wdl_accuracy", "wdl_ce"):
            values = [(row["swa"] or row["final"])[key] for row in selected]
            summary[key + "_mean"] = statistics.mean(values)
            summary[key + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        means.append(summary)
    report = {"config": spec, "train_sha256": train.fingerprint, "validation_sha256": validation.fingerprint,
              "train_positions": len(train), "validation_positions": len(validation), "seeds": list(seeds),
              "sources": train.sources, "encoding": train.encoding,
              "scope": "local pilot; data and compute differ from paper; inspect sources before interpreting accuracy",
              "runs": rows, "summary": means}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--allow-missing-aux", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    report = ablate(args.config, args.train, args.validation, args.out, args.seeds, args.allow_missing_aux)
    print(json.dumps(report["summary"], indent=2))
