# Architecture and replication fidelity

[Back to README](../README.md)

| Component | Current implementation |
|---|---|
| Input | 64 square tokens with 112 features; mover-relative pieces and up to seven historical positions, following the selected input format |
| Encoder | Bidirectional attention, Post-RMSNorm, DeepNorm initialization/residual scaling, Mish feedforward layers |
| Position ablations | Learned absolute offsets, 225 displacement-tied relative biases, Shaw relative query/key/value vectors |
| Policy | Shared policy embedding, separate main/soft attention heads, 1,858 actions with explicit promotion handling |
| Value | Separate result WDL, search-value `q`, and short-term `st` heads; scalar, error, and optional categorical outputs |
| Objective | Legal-masked policy CE; temperature-4 auxiliary target weighted by 8; WDL CE; value MSE; detached squared-error targets; optional categorical CE |
| Training | NAdam `(0.9, 0.98)`, epsilon `1e-7`, gradient clipping at 10, gradient accumulation, SWA, atomic checkpoints |
| Native data | V6 and authors' V7 game chunks with legal-move chronology checks; sparse policies and packed planes in SQLite; game splits and exact-input deduplication |
| Inference | Highest legal policy move, or one-ply child WDL evaluation; unused training heads are skipped, with mover-perspective reversal and exact terminal outcomes |
| Evaluation | Policy CE/KL and move agreement, WDL CE/accuracy, auxiliary MSE, ChessBench puzzle scoring and paired opening-book match harness |

The implementation is guided by Sections II–III and Appendices A–B of the [paper](https://arxiv.org/pdf/2409.12272). The authors' trainer is pinned to [7b3ee3f](https://github.com/daniel-monroe/lczero-training/tree/7b3ee3fcdb49d8abeb034718935b099134c11882), and the inference reference to [Lc0 96f98af](https://github.com/LeelaChessZero/lc0/tree/96f98aface79bb548cb6d2f2877f69c699d3e67a). These are default-branch snapshots selected through the end of the v2 publication day; the engine commit is later than the submission timestamp that day. They are not verified provenance for every paper checkpoint. File paths, commits, and verified Git blob hashes are recorded in [results/upstream-sources.json](../results/upstream-sources.json). The authors' example YAML is not an identified configuration for the reported large model.

### Fidelity decisions and unresolved differences

- **Equation versus upstream code.** `shaw_cross_term=true` includes the `aQ · aK` term obtained by expanding Equation 6. Setting it to `false` omits that term, matching the additive relative-query/key calculation in the inspected upstream code. Both settings have independent equation and gradient tests. The pilot uses `true`.
- **DeepNorm.** `deepnorm="canonical"`, used by the archived pilots, multiplies the residual by `(2L)^(1/4)` and applies Xavier standard-deviation gain `beta=(8L)^(-1/4)`. `deepnorm="trainer"` instead scales the branch by the reciprocal residual multiplier and uses the pinned trainer's truncated-normal initialization with **variance** multiplier beta. This raises the affected weights' initial variance by `1/beta`: 2.83× for eight layers and 3.31× for fifteen. The latter option also reproduces fixed-seed, same-shape initializer reuse. It matches the Keras distribution and the source's RMS epsilon placement, not TensorFlow's random draws. Numerical parity with a TensorFlow runtime is not established.
- **Input/action format.** The archived PGN pilots use `cf112-v1`: rank-mirrored Black, ordinary king-destination castling, and all available recent history. Native training uses separate `lc0-training-v1:<format>` checkpoints with Lc0 canonical transforms, history rules, and king-to-rook castling indices. Inference selects the encoder from checkpoint metadata. Native support covers standard chess; Chess960 inference and pretrained Lc0 weight conversion remain unsupported.
- **Parameter counts.** The dimensions listed in the paper give the counts below when attention width equals token width. The trainer exposes those widths independently, now supported by `attention_dim` (default: `d_model`). A hypothetical attention width of 1,536 raises the large model from 205,656,679 to 242,297,959 parameters, but no retrieved configuration establishes that choice. The large-model discrepancy remains unresolved; the default large configuration is unchanged. Counts by component are in [results/model-conformance.json](../results/model-conformance.json).
- **Training choices.** SWA start/frequency, 128 categorical buckets for the large model, float32 precision, microbatch sizes, sampling with replacement, and PyTorch NAdam are explicit local choices. These are not all specified in the paper. Distributed training remains unimplemented. Native loading now bounds memory per game/batch, but large-corpus throughput and GPU execution remain unmeasured.

| Configuration | Layers / width / heads / FF width | Actual parameters | Validation performed |
|---|---|---:|---|
| `configs/pilot.json` | 2 / 64 / 4 / 64 | 223,967 (Shaw) | Nine full local runs across three encodings and three seeds |
| `configs/cf6m.json` | 8 / 256 / 8 / 256 | 5,743,463 | Forward/backward with finite gradients |
| `configs/cf6m-trainer.json` | 8 / 256 / 8 / 256 | 5,743,463 | Trainer-style DeepNorm/additive Shaw; CPU forward/backward and small-model reference checks |
| `configs/cf240m.json` | 15 / 1024 / 32 / 4096 | 205,656,679 | Shape/parameter inspection on PyTorch's meta device only |

The paper reports approximately 6M and 243M parameters. Its small-model recipe uses batch 2048 and LR changes at 1.6M/1.8M updates; its large recipe uses batch 4096 and 3.7M updates, with LR changes at 3.2M/3.6M. The configuration files retain those scales. They are specifications for future experiments, not claims that those runs happened. [Paper, Section III](https://arxiv.org/pdf/2409.12272)

`configs/cf6m-trainer.json` combines the paper's small-model tasks/schedule with the pinned source's additive Shaw and initialization/residual mechanics. It is an explicit alternative to the equation-based configuration, not a recovered experiment configuration. Optimizer trajectories, TensorFlow random seeds, and complete checkpoint conversion remain unverified. The NumPy reference covers old-style embedding, Shaw attention, and the stated heads; it does not cover the source's optional new embedding, smolgen, quantization, extra heads, or unconstrained 64×64 logit gates.

## Remaining research work

1. Identify the exact experiment configuration/checkpoints within or beyond the pinned sources, reconcile the 243M parameter count and initialization/attention ambiguities, and establish cross-framework numerical parity where possible.
2. Obtain authentic Lc0 self-play chunks from the relevant periods and validate the implemented native importer against them, including chronology, rescoring, empty-history settings, and corpus throughput. Generated format checks and a CCRL played-move pilot are insufficient evidence.
3. Train the intended model scales and position ablations with the required data and compute. Validate GPU execution, the chosen precision, optimizer behavior, and distributed training before launching long jobs. Full-scale training also requires a defined compute budget.
4. Run the paper's held-out policy/value evaluation, curated full-line puzzle benchmark, and paired opening-book games against its reference agent. Report uncertainty, evaluation FLOPS, and actual measured playing strength; these cannot be inferred from unit tests or move agreement.
5. Compare any proposed accuracy improvements on matched data/compute across seeds. Benchmark attention changes on the target GPU and report regressions as well as gains. Investigate the attention-map and positional examples only after a sufficiently trained model exists.

The target is **arXiv:2409.12272v2**. Later BT4/Leela-CF architectures and weights are not substitutes for its CF-240M checkpoint. The [replication audit](../results/replication-audit.json) records the outstanding requirements and historical asset-access checks.

