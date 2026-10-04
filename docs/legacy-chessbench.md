# Legacy ChessBench implementation

[Back to README](../README.md)

This document describes the original top-level `common.py`, `model.py`,
`label.py`, `train.py`, `launch.py`, and `extract.py`. These scripts target
*Grandmaster-Level Chess Without Search* and are separate from the
`chessformer/` package. Their historical results and Modal setup are preserved
below; they are not results for the current replication study.

A searchless chess transformer: a small model that predicts Stockfish's action-values
directly from a board position, with no tree search at inference time. Reimplements
the setup from [Grandmaster-Level Chess Without Search](https://arxiv.org/abs/2402.04494)
(ChessBench), scaled down to a single-GPU budget and run on Modal.

## Approach

The model learns the **action-value function** `Q(s, a)`: given a board state `s` and a
legal move `a`, predict the win probability after playing that move. At inference the
policy scores every legal move and plays the argmax — no search.

Following the paper:

- **Board encoding.** A FEN string is flattened to a fixed **77-token** sequence
  (64 board squares with run-length encoding expanded, 1 side-to-move, 4 castling,
  2 en passant, 3 halfmove clock, 3 fullmove number). Vocabulary is 32 characters.
- **Action encoding.** Moves index into a lexicographically sorted array of the
  **1968** possible UCI moves. The action is appended as a 78th token.
- **Value target.** Stockfish's centipawn score is converted to a win probability with
  `win% = 1 / (1 + exp(-0.00368208 * cp))`, then discretized into **128 bins**.
- **Loss.** HL-Gauss: instead of a one-hot bin, the target is a Gaussian centered on the
  true win probability with `sigma = 0.75 / K`, integrated over each bin. This turns
  regression into classification while keeping the ordinal structure of the value scale.

## Pipeline

| Stage | Command | What it does |
|---|---|---|
| Boards | `modal run label.py --stage boards` | Parse CCRL PGNs, sample and dedupe positions, write 64 shards |
| Label | `modal run --detach label.py --stage label` | 64 containers run Stockfish over every legal move |
| Pack | `modal run train.py --stage pack` | Concatenate shards into flat arrays |
| Train | `modal run --detach train.py --stage train` | bf16 training on one A100 |
| Eval | `modal run train.py --stage eval` | Top-move match against the Stockfish oracle |

Data comes from the CCRL engine-game archive already on the `cclr-data` Modal volume.
Positions are deduplicated on the first four FEN fields, so the move counters do not
make transpositions look like distinct boards.

### Labeling

The expensive stage. Every legal move of every board gets its own Stockfish search at
20ms, which is what makes the labels an *action-value* dataset rather than a
state-value one. Measured throughput is ~22ms per action-value including engine
overhead, so ~54M action-values is on the order of 330 single-threaded hours. Spread
over 64 serverless containers that lands near 6 hours of wall clock.

Two operational details matter at this scale:

- **Run detached.** With a plain `modal run`, the local client holds a heartbeat to the
  app; if that gRPC call fails, Modal stops the app and cancels every container
  mid-flight. `--detach` decouples the run from the local process.
- **Checkpoint and resume.** Each shard saves progress every 2000 boards and restarts
  from its last checkpoint, so a preemption or cancellation costs minutes, not hours.
  Stockfish is also restarted in-process if it dies on a position.

Note that Debian installs the engine to `/usr/games/stockfish`, which is not on `PATH`.

## Model

A 6.1M-parameter decoder-only transformer: 7 layers, `d_model` 256, 8 heads, SwiGLU
feed-forward (hidden 688), post-normalization, learned positional embeddings. The bin
logits are read from the final (action) token position.

Attention uses `F.scaled_dot_product_attention` with `is_causal=True`. This matters for
more than speed: `nn.MultiheadAttention` with an explicit float mask materializes a
`[batch, heads, 78, 78]` score matrix per layer and saves it for backward, which runs
an A100 out of memory at batch 4096. The fused path never materializes it.

## Training

10 epochs over the labeled action-values, batch 4096, AdamW at 3e-4 with linear warmup
and cosine decay, bf16 autocast plus `torch.compile`, on one A100. The HL-Gauss targets
are computed on-GPU per batch from the stored win probabilities rather than
precomputed, since a materialized `[N, 128]` float target array would be far larger
than the labels themselves.

## Evaluation

**Action accuracy** is the fraction of held-out boards where the move the model scores
highest is the move Stockfish scores highest, measured over 50k boards from games in a
disjoint split of the archive.

### The oracle is noisy, and that bounds the metric

Re-labeling stored positions with a second, independent Stockfish run at the same 20ms
budget reproduces the *values* closely — mean absolute win-probability difference of
**0.0103** — but picks the same *top move* only **65%** of the time.

The two facts are consistent: on most boards several moves are nearly equal in value, so
tiny evaluation differences reshuffle the argmax without meaningfully changing the
position's assessment. It does mean action accuracy is measured against a target that
disagrees with itself about a third of the time, so the achievable ceiling is well below
100% and the headline number should be read with that in mind. The values the model
actually trains on are far more stable than the argmax computed from them.
