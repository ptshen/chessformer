# Experiments and measured results

[Back to README](../README.md)

## Position-encoding pilot

The dataset contains 1,024 training positions from 128 CCRL games and 256 validation positions from 32 different games. Every run uses 200 optimizer updates, effective batch 32, microbatch 16, and five SWA snapshots at steps 160–200. Common weights and sampling are seeded identically across variants; the architectural parameter counts differ.

Results below are **SWA means ± sample standard deviation across seeds 0, 1, 2**. These are not confidence intervals, Elo estimates, or the paper's held-out self-play metrics.

| Position representation | Parameters | Played-move agreement | Policy KL | WDL accuracy |
|---|---:|---:|---:|---:|
| Absolute | 145,759 | 15.49% ± 2.66 pp | 3.0073 ± 0.0546 | 34.38% |
| Relative bias | 139,367 | 15.63% ± 2.44 pp | 3.0076 ± 0.0553 | 34.51% |
| Shaw | 223,967 | 15.76% ± 2.60 pp | 3.0090 ± 0.0557 | 34.77% |

Shaw's 0.26 percentage-point agreement advantage over absolute positions is smaller than the seed variation; its policy KL is slightly worse. This small experiment validates the workflow and does **not** establish a playing-strength improvement or reproduce the paper's ablation ranking. WDL learning is also weak on this tiny dataset.

Machine-readable evidence is in [results/position-ablation.json](../results/position-ablation.json), including all nine runs, initial/final/SWA metrics, data hashes, source hashes, and checkpoint hashes. Run manifests and logs are retained in [results/pilot-runs](../results/pilot-runs); local model files are under `runs/position-ablation-v2/` and excluded from version control. The dataset recipe and hashes are in [results/pilot-data-manifest.json](../results/pilot-data-manifest.json). These runs predate the native-data implementation and retain their original source hash; they were not rerun or relabeled as native-data experiments.

```sh
python3 -m chessformer.experiments --config configs/pilot.json \
  --train data/pilot-v1/train.jsonl --validation data/pilot-v1/validation.jsonl \
  --out runs/new-position-ablation --seeds 0 1 2 --allow-missing-aux
```

## Improvement investigated: factorized relative attention

The dense implementation gathers three `[heads,64,64,head_dim]` relative tensors. The alternative projects Q/K against the 225 displacement vectors, gathers the appropriate logits, and scatter-sums attention weights by displacement before multiplying relative values. It computes the same equations while trading the gathered tensors for batch-dependent intermediates.

The CPU microbenchmark uses eight heads of depth 32, float32, three warmups, and median timing across 20 iterations. Times cover the attention operation, not a complete model or GPU workload.

| Batch | Dense forward | Factorized forward | Dense forward+backward | Factorized forward+backward |
|---:|---:|---:|---:|---:|
| 1 | 1.939 ms | 0.304 ms | 5.705 ms | 0.919 ms |
| 16 | 4.649 ms | 3.034 ms | 12.692 ms | 8.350 ms |
| 32 | 6.106 ms | 7.056 ms | 16.736 ms | 18.102 ms |

At batch 1, factorization was **6.4× faster for forward** and **6.2× for forward+backward** on this CPU. At batch 32 it was slower. The maximum float32 output discrepancy across these benchmark inputs was `2.15e-6`; float64 tests check outputs and gradients against direct equations. This is an implementation improvement for the measured small batches, not a strength improvement. Neither GPU speed nor peak memory has been measured. Keep the choice explicit via `--attention-impl dense|factorized`; there is no universal automatic dispatch rule.

```sh
python3 -m chessformer.benchmark --out results/new-attention-benchmark.json
```

Raw timings and environment: [results/attention-benchmark.json](../results/attention-benchmark.json).

## Improvement measured: skip unused inference heads

`choose_move` now asks `Chessformer` for just the policy or result-WDL head. This skips the soft-policy and auxiliary-value heads; a value agent also skips the policy embedding. `model(inputs)` still computes every training head. Explicit selection is available as `model(inputs, heads=("policy",))` or `heads=("wdl",)`, and supports multiple heads plus attention maps. Model parameters and checkpoint layouts are unchanged. Custom modules implementing only `forward(inputs)` still work through the original path.

Requested logits match the all-head path **exactly** in the measured CPU float32 cases. Tests also check input gradients, both mover orientations/encodings, castling, EP, promotion, terminal children, and move-choice agreement. The NumPy full-model check still passes after this change. Skipping heads changes computation, not the learned policy or evaluation.

The following timings include board/history encoding and complete move selection from one fixed ten-ply opening. Measurements use CPU float32, four threads, five warmups, and 30 paired rounds with randomized condition order. The pilot uses its archived trained weights and factorized attention; CF-6M uses the trainer profile with seeded **untrained weights and nonzero relative tables**, using dense attention. Thus these are implementation benchmarks, not playing-strength results.

| Model / strategy | All heads | Selected head | Measured speedup |
|---|---:|---:|---:|
| Pilot policy | 0.917 ms | 0.816 ms | 1.124× |
| Pilot value | 10.486 ms | 9.452 ms | 1.109× |
| CF-6M policy | 17.272 ms | 16.976 ms | 1.017× |
| CF-6M value | 67.728 ms | 64.298 ms | 1.053× |

The raw reports include paired bootstrap intervals and every timing sample. These intervals describe within-process timing variation, not variability across machines or sessions. Some forward-only cases show little or no speed gain: CF-6M dense policy at batch 1 measured 17.134 ms with all heads and 17.171 ms with selection. GPU latency and peak memory remain unmeasured. All parameters remain resident, so this change does not shrink checkpoint storage.

The full-model benchmark also compares the existing dense/factorized implementations. For CF-6M policy at batch 1, they measured 17.171/4.289 ms; at batch 20, 54.745/50.388 ms. These implementation variants were measured in separate blocks, so those ratios have weaker timing controls than the interleaved head-selection comparison. Their largest output difference was `2.39e-6`, while changing only the selected heads had zero difference. Factorization performs more counted matrix arithmetic despite its lower latency in these cases.

For compute accounting, PyTorch's registered matrix-operation counts agree with an independent calculation from matrix dimensions. **One multiply plus one add counts as two FLOPs** here. The counts omit nonlinearities, normalization, elementwise operations (including the Shaw cross term), gathers/scatters, encoding, and memory traffic. They are not complete hardware instruction counts, and the paper's FLOPS convention has not been established as equivalent.

| Local configuration, dense, batch 1 | All-head matrix FLOPs | Policy-only matrix FLOPs | WDL-only matrix FLOPs |
|---|---:|---:|---:|
| CF-6M trainer profile | 541,329,152 | 517,603,328 | 491,782,912 |
| Current large configuration | 25,506,154,240 | 25,214,582,784 | 24,807,736,064 |

The CF-6M reductions are 4.4% for policy and 9.2% for WDL. The large row is a **shape-only meta-device count** for the current 205.7M-parameter configuration; it is not a runtime test or a resolution of the paper's 243M parameter discrepancy. Child-value cost depends on the number of nonterminal children actually evaluated, which the match harness records, rather than always being 20 evaluations.

```sh
python3 -m chessformer.inference_benchmark \
  --checkpoint runs/position-ablation-v2/shaw-seed0/model.pt \
  --out results/new-inference-pilot-benchmark.json
python3 -m chessformer.inference_benchmark --config configs/cf6m-trainer.json \
  --out results/new-inference-cf6m-benchmark.json
```

Evidence: [pilot timings](../results/inference-pilot-benchmark.json), [CF-6M timings](../results/inference-cf6m-benchmark.json), [arithmetic counts and conventions](../results/inference-arithmetic.json), [updated model-equation check](../results/inference-model-conformance.json).
