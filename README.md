# Chessformer

An independent PyTorch implementation of **[Mastering Chess with a Transformer Model](https://arxiv.org/abs/2409.12272v2)** by Daniel Monroe and Philip A. Chalmers, with reproducible CPU experiments and inference benchmarks.

## What's implemented

- A bidirectional chess transformer with 64 square tokens, 112 input features, a 1,858-action policy, and result/search-value heads.
- Absolute, relative-bias, and Shaw position representations; dense and factorized relative attention.
- NAdam training with gradient accumulation, stochastic weight averaging, and resumable checkpoints.
- Native Lc0 V6 and authors' V7 data ingestion, game chronology validation, and disk-backed datasets.
- Policy and one-ply value move selection, full-line puzzle scoring, and paired opening-book matches.

Architecture choices and differences from the paper are documented in [Architecture and replication fidelity](docs/implementation.md).

## Quick start

Use Python 3.11 or newer and run commands from the repository root.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

A checkout without the optional upstream sources runs 96 tests and skips two reference checks. The archived validation passed all **98 tests** with those sources present. See [Validation and evaluation](docs/validation-and-evaluation.md) for fetching the pinned sources and running the reference checks.

To run an attention benchmark without a dataset or checkpoint:

```sh
python -m chessformer.benchmark --out results/my-attention-benchmark.json
```

### Train a small pilot

Supply separate PGN games as `cclr/train/*.pgn` and `cclr/test/*.pgn`. The directory spelling `cclr` follows the original local setup. These pilots learn played moves and game outcomes; they do not have the paper's search-policy or auxiliary value targets.

```sh
python -m chessformer.data --pgn-root cclr --out data/pilot-v1 \
  --train-games 128 --validation-games 32 --positions-per-game 8

python -m chessformer.train --config configs/pilot.json \
  --train data/pilot-v1/train.jsonl --validation data/pilot-v1/validation.jsonl \
  --out runs/my-pilot --allow-missing-aux

python -m chessformer.play --checkpoint runs/my-pilot/model.pt --strategy policy
```

Use a new output directory for each experiment. See [Data and training](docs/data-and-training.md) for the input schema, native self-play importer, checkpoint resume, and inference with game history.


[Experiments and measured results](docs/experiments.md) includes uncertainty, regressions, reproduction commands, and links to the raw reports. These measurements establish local implementation behavior, not chess strength.

## Model configurations

| Configuration | Layers / width / heads / FF width | Parameters | Status |
|---|---|---:|---|
| [Pilot](configs/pilot.json) | 2 / 64 / 4 / 64 | 223,967 | Trained in the local Shaw pilots |
| [CF-6M](configs/cf6m.json) | 8 / 256 / 8 / 256 | 5,743,463 | Forward/backward checked |
| [CF-6M trainer profile](configs/cf6m-trainer.json) | 8 / 256 / 8 / 256 | 5,743,463 | Alternative upstream initialization/attention mechanics checked |
| [Large model](configs/cf240m.json) | 15 / 1024 / 32 / 4096 | 205,656,679 | Shape inspection only; paper's reported 243M count unresolved |

Full replication still requires authentic self-play and benchmark data, the original large-model configuration or an explanation of its dimensions, and GPU training/evaluation. Distributed training is not implemented. The [research requirements](docs/implementation.md#remaining-research-work) and [replication audit](results/replication-audit.json) detail the outstanding work.

## Repository guide

| Path | Contents |
|---|---|
| [`chessformer/`](chessformer/) | Current model, data pipeline, training, evaluation, and benchmarks |
| [`configs/`](configs/) | Pilot and paper-scale experiment specifications |
| [`tests/`](tests/) | Mathematical, chess, data, training, and evaluation checks |
| [`results/`](results/) | Archived metrics, manifests, timings, source hashes, and validation reports |
| [`tools/`](tools/) | C++ harness for the optional Lc0 encoder comparison |
| [`docs/`](docs/) | Detailed methods, usage, results, and limitations |

The top-level `model.py`, `train.py`, and related scripts belong to an earlier implementation of *Grandmaster-Level Chess Without Search*. Their [legacy documentation](docs/legacy-chessbench.md) is retained separately. Use `python -m chessformer...` for this project.

Datasets, downloaded upstream code, checkpoints, and local settings are excluded by `.gitignore`. Pinned upstream revisions and file hashes are recorded in [upstream-sources.json](results/upstream-sources.json); downloaded reference files retain their upstream notices.
