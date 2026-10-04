# Validation and evaluation

[Back to README](../README.md)

The recorded 2026-10-01 validation passed **98 tests**, including the optional source-reference checks. Current source hashes and verification results are in [results/validation.json](../results/validation.json). The earlier 29-test audit of the archived PGN pilots is preserved in [results/pilot-validation.json](../results/pilot-validation.json). Two optional source-reference tests require the downloaded pinned sources and are skipped when they are absent; run the source fetch below to include them.

`python3 -m pytest -q` covers direct Shaw equations and gradients, two-dimensional displacement indexing, all three position representations, mixed-precision gradients, legal masking, promotions for both colors, castling, en passant, history orientation, repetition, input/game split leakage, auxiliary-gradient detachment, microbatch equivalence, exact CPU restart including SWA, child-value sign, terminal checkmate, and complete-line puzzle scoring. Native tests additionally cover binary validation, V6 EMA semantics, sparse disk round trips, interrupted-import isolation, native training/restart with all auxiliary targets, and checkpoint-selected inference.

Evaluation tests cover PGN history, alternative immediate mates, malformed/duplicate puzzles, paired colors, adjudication sign and strict score threshold, draw claims, PGN output, and exclusion of incomplete matches from strength estimates.

Independent reference checks supplement the tests:

- **Binary decoding:** 56 generated records across seven input formats match the authors' pinned decoder. Input planes and WDL match exactly; policy, Q, ST, and EMA discrepancies are at most `2.98e-8`. All 1,858 policy entries match the authors' ordered list. [Raw report](../results/native-conformance.json).
- **Position encoding and actions:** the Python encoder matches compiled, unmodified publication-era Lc0 chess/encoder sources across 495 board histories × seven formats = **3,465 cases**, covering all eight transforms. Trainer-scaled planes match exactly, as do **80,633 legal move indices**. Cases include generated legal histories, pawnless positions, castling, promotion, and EP. [Raw report](../results/native-encoder-conformance.json).

These checks use generated fixtures, not authentic self-play records or trained reference weights. They establish the tested format behavior, not model accuracy or complete TensorFlow/PyTorch parity. A newer engine snapshot gave a move-index transform discrepancy on a pawnless position; the oracle therefore explicitly uses the publication-era revision above.

The current-position decoder used for chronology validation also matches **3,465 positions and 80,633 legal move indices** from the compiled encoder. Its frames are constructed from that encoder's output, not the Python encoder. All eight canonical transforms, both colors, castling, promotion, and EP are covered. Separate stream tests reject two missing plies across all seven formats and both V6/V7, wrong played moves, changed clocks, and overlapping piece planes; valid castling, EP, promotions, and retained deleted frames pass. A contiguous subline deliberately passes to document the endpoint limitation. [Chronology decoder report](../results/native-chronology-conformance.json).

A separate **model-equation check** compares eight cases spanning float32/float64, dense/factorized attention, and equal/unequal token and attention widths. The independent NumPy implementation uses the pinned trainer's displacement and policy-map builders and nonzero relative tables. All heads, promotion logits, and attention probabilities agree within `1.49e-6` in float32 and `1.78e-15` in float64. Selected full-model input derivatives agree with finite differences. A 786,432-weight initializer sample has variance `0.000337585` against the requested `0.000337207` and respects the truncation bound. This is a source-guided NumPy translation, **not execution of TensorFlow**. [Current model report](../results/inference-model-conformance.json), [earlier model report](../results/model-conformance.json), [CF-6M trainer-profile smoke check](../results/cf6m-trainer-smoke.json).

```sh
# Fetch/check sources by recorded Git blob hashes, retaining upstream notices.
python3 -m chessformer.reference --fetch --verify-native \
  --out results/new-native-conformance.json

# Optional compiled oracle: requires C++20, protoc, and protobuf headers.
# Omit --include-dir if headers are already on the compiler's search path.
python3 -m chessformer.engine_reference --include-dir /opt/homebrew/include \
  --out results/new-native-chronology-conformance.json

# Source-guided full-model equations and component-level parameter counts.
python3 -m chessformer.model_reference --out results/new-model-conformance.json

# Execute the pinned ChessBench evaluator functions on synthetic cases.
python3 -m chessformer.evaluation_reference --out results/new-evaluation-conformance.json
```

The evaluation reference is [ChessBench commit 4a41136](https://github.com/google-deepmind/searchless_chess/tree/4a41136164cbf79786eea628513579313e1fe454), the last default-branch commit before the paper's v2 date. **29 puzzle cases and 24 match cases** agree with its unmodified evaluator functions, including agent input histories, colors, results, and final positions. The oracle uses scripted agents and adjudication scores; it does not test a Stockfish binary, BayesElo, GC-270M weights, or an authentic benchmark. [Protocol report](../results/evaluation-conformance.json). All **59 pinned source files** pass their recorded Git blob checks.

## Puzzles

Puzzle evaluation accepts the curated ChessBench CSV with **PGN history**, or a local [Lichess CSV](https://database.lichess.org/#puzzles) with FEN. It applies the first listed move as the opponent's setup, then checks every solver move. Following ChessBench, an alternative move that immediately checkmates is accepted. `--scoring exact` disables that exception. PGN columns select ChessBench input automatically; `--input-format chessbench` explicitly requires that history. Malformed lines, PGN/FEN disagreement, and duplicate IDs raise errors. Reports include data/checkpoint hashes, per-puzzle outcomes, aggregate accuracy, 100-point rating bins, and descriptive Wilson 95% intervals. Arbitrary Lichess CSVs are not the paper's curated benchmark.

```sh
python3 -m chessformer.puzzles --checkpoint runs/my-pilot/model.pt \
  --csv /path/to/puzzles.csv --input-format chessbench \
  --strategy policy --out results/puzzles.json
```

## Paired matches

The match harness samples 500 opening-book entries without replacement using NumPy seed 1 and plays both colors, yielding 1,000 games by default. It preserves opening history, stops on current threefold repetition or a claimable fifty-move draw, and uses the reference's Stockfish adjudication: 0.01 seconds per position, default UCI options, any mate score or **absolute score strictly greater than 1,300 centipawns**. Exact terminal outcomes take precedence. Supply the intended Stockfish 16 executable explicitly; the harness records its reported ID and binary hash but does not install it or enforce its version. `--protocol rules` omits engine adjudication and changes the benchmark protocol.

```sh
# Local-checkpoint comparison; use new output directories for each match.
python3 -m chessformer.matches --checkpoint runs/candidate/model.pt \
  --opponent-checkpoint runs/reference/model.pt --opponent-name reference \
  --openings /path/to/eco_openings.pgn --adjudicator /path/to/stockfish-16 \
  --games 1000 --seed 1 --out results/match-candidate-reference
```

`--opponent-uci /path/to/engine --opponent-time 0.05` supports a timed UCI opponent. For the paper's GC-270M comparison, `--opponent-factory searchless_chess.src.engines.constants:ENGINE_BUILDERS:270M` can call the pinned baseline's builder. That requires a separate importable ChessBench checkout, its dependencies and downloaded weights, and its expected working directory (`src`, because it resolves checkpoints relative to the current directory). The builder selects training step 6,400,000. This adapter has not been run with those weights and does not verify external checkpoint provenance; use absolute paths for the candidate, opening book, and outputs when changing directories.

Matches save PGN, per-game JSONL, a manifest, and a summary. `--bayeselo /path/to/bayeselo` runs `readpgn`, `elo`, `mm`, `exactdist`, and `ratings` on a completed match and saves raw output; the real executable has not been tested here. Without it, the report contains a descriptive score-to-logistic-Elo conversion and a conservative Hoeffding 95% score interval using opening pairs as sampling units. This is not BayesElo or an absolute Elo rating. Games reaching `--max-plies` are recorded as incomplete, never draws; any incomplete game or color pair suppresses the score estimate.

A two-game **CLI smoke run**, using the archived pilot models and a synthetic opening, exercised policy/value inference and PGN export. It counted 8 evaluated positions for 8 policy decisions and 247 positions for 8 batched child-value decisions. Both games deliberately stopped after eight additional plies, leaving **no score or strength estimate**. [Smoke manifest](../results/evaluation-smoke/manifest.json), [output](../results/evaluation-smoke/result.json), [opening fixture](../results/evaluation-smoke-openings.pgn). This archived run predates head selection and evaluated all heads. Position counts alone are not FLOPS; current inference skips unused heads as described above.

No official puzzle or match evaluation has been run. The pinned download script identifies [curated puzzles](https://storage.googleapis.com/searchless_chess/data/puzzles.csv) and [ECO openings](https://storage.googleapis.com/searchless_chess/data/eco_openings.pgn). Attempts to retrieve them on 2026-10-01 failed in this environment (shell DNS unavailable; browsing also could not retrieve the assets). Their contents and hashes remain unverified, and neither Stockfish nor BayesElo is installed on PATH.

Use `--strategy value` for child-value selection. Encoder attention probabilities are also available through `model(inputs, return_attention=True)` as `[batch,layer,head,query,key]`; interpretation on a trained full-scale model remains future work.
