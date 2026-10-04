# Data and training

[Back to README](../README.md)

Python 3.11+ is required. The recorded experiments used Python 3.13, PyTorch 2.9.1, NumPy 2.4.4, python-chess 1.11.2, CPU float32, and four threads on macOS ARM64. These are the versions used for the archived runs; CUDA and MPS were unavailable in that environment.

```sh
python3 -m pip install -e '.[test]'
python3 -m pytest -q

# Supply separate PGN games under cclr/train/ and cclr/test/ first.
python3 -m chessformer.data --pgn-root cclr --out data/pilot-v1 \
  --train-games 128 --validation-games 32 --positions-per-game 8

python3 -m chessformer.train --config configs/pilot.json \
  --train data/pilot-v1/train.jsonl --validation data/pilot-v1/validation.jsonl \
  --out runs/my-pilot --allow-missing-aux

python3 -m chessformer.play --checkpoint runs/my-pilot/model.pt --strategy policy
python3 -m chessformer.play --checkpoint runs/my-pilot/model.pt --strategy value
```

Run these commands from the repository root. Datasets and checkpoints are excluded from version control. The PGN importer expects `*.pgn` files directly inside `cclr/train/` and `cclr/test/`; place disjoint games in those directories before running it.

Choose a new output directory if a dataset or run already exists. To resume an interrupted run, repeat its training command with `--resume`. `--stop-after 100` stops after an optimizer update without changing the configured schedule. Checkpoints store model, optimizer, SWA, step, sampler RNG, CPU/CUDA RNG, source hash, configuration, and dataset hashes; changing those inputs rejects resume. `model.pt` contains the SWA inference weights when available; `checkpoint.pt` retains training state. Bit-exact CPU restart is tested; cross-device equivalence is not claimed.

A FEN alone cannot recover game history. For inference with history, pass `--fen` for the initial position and `--moves e2e4 e7e5 ...` for the subsequent moves.

## Data contract

The JSONL loader is intended for local pilots. Each line specifies an initial FEN, legal move history, game ID, a policy over legal UCI moves, WDL in **current-side-to-move order `[win, draw, loss]`**, and `q`/`st` in `[-1,1]` when available. Policy weights can be counts or normalized probabilities; they are normalized on loading. Legal moves with zero target mass still participate in the softmax denominator.

Example schema, with illustrative values rather than measured labels:

```json
{"encoding":"cf112-v1","game_id":"example","moves":["e2e4"],"policy":{"e7e5":60,"c7c5":40},"wdl":[0,1,0],"q":0.1,"st":0.05}
```

Omitting `initial_fen` means the standard starting position. `q` is a supplied search reward; `st` is a supplied short-term reward estimate. The loader does not fabricate these from game outcomes. Missing auxiliary targets fail by default. `--allow-missing-aux` masks them explicitly and records their availability in the manifest.

The pilot's 112 input channels are defined in [encoding.py](../chessformer/encoding.py): eight groups of 12 piece planes plus repetition, castling-rook masks at 104–105, zeros at 106–107, en-passant file on rank 8 at 108, halfmove clock/100 at 109, zero at 110, and ones at 111. Unavailable history is zero-filled. This remains the historical pilot format; the native conventions below are separate.

The PGN bootstrap uses the **played move as a one-hot policy and the final game result as WDL**. It does not contain the paper's search distributions, search values, or short-term targets. It skips the first 16 plies, samples up to eight positions per game, hashes full move sequences for game identity, and rejects overlapping games and identical 64×112 inputs across splits. This protects this pilot from exact input leakage without claiming that related positions in different games are statistically independent.

### Native Lc0 chunks

[native.py](../chessformer/native.py) reads complete, chronological `.gz` games, individually or inside `.tar`, `.tar.gz`, `.tgz`, or `.tar.bz2` archives. V6 records are 8,356 bytes; supported V7 means the authors' specific 8,396-byte rescored format, not every later format bearing that version number. Input formats `1, 2, 3, 4, 5, 132, 133` are supported for **standard chess** and must be selected explicitly. Mixed versions/formats within a game, truncated records, invalid targets, oversized games, and non-alternating movers raise errors.

[native_validation.py](../chessformer/native_validation.py) additionally reconstructs each current position, reverses the stored canonical transform, and replays its recorded `played_idx`. The next frame must match the resulting pieces, mover, castling rights, en-passant state, and halfmove clock. This catches missing pairs of plies that a mover-alternation check misses, before they corrupt the future-reward EMA. Legacy formats omit EP, so the validator carries it through replay and can infer an initial double pawn push from the preceding history slot. Illegal recorded moves and overlapping piece planes are rejected. Deleted frames still undergo chronology checks. **This verifies interior transitions, not whether a chunk's beginning or end was removed.** Complete, unsampled game provenance remains a source requirement; stored history beyond the EP inference is not reconstructed by this check.

Policies retain zero-visit legal actions and treat `-1` as the illegal sentinel. WDL comes from `result_q/result_d`; the search target is `root_q`. **V6's final float is not a short-term target.** Before sampling, V6 short-term targets are calculated as `st[i] = q[i]/6 - 5*st[i+1]/6`, ending with `st[last] = q[last]`, matching the authors' future-reward EMA. V7 supplies its stored short-term value. Deleted frames contribute to chronology but are excluded as training samples.

[native_data.py](../chessformer/native_data.py) writes a finalized SQLite dataset atomically. It holds at most one game in preprocessing memory (default cap: 8,192 plies), packs board bits, stores sparse legal policies, and loads only requested batches. Each SQLite connection has a 32 MiB page cache. Game identities exclude rescoring labels; deterministic game splits and a unique feature hash prevent identical inputs crossing splits. Duplicate inputs are discarded, which changes their training frequency and is an explicit local sampling choice. The database retains per-game source paths and content hashes; its manifest records split counts, duplicates, import settings, and the chronology check, with the final database hash in a sidecar. The dataset schema is now `chessformer-native-sqlite-v2`; older databases require reimport so they cannot bypass that check. Training source fingerprints also include the validator, so old training-state resumes are rejected after this change; inference checkpoint formats remain compatible. Opening a dataset hashes the complete database; large-corpus preprocessing and random-read throughput have not been benchmarked.

For data acquisition, an Lc0 maintainer identifies the [T80 archive](https://storage.lczero.org/files/training_data/test80/) as the source for BT3/BT4 training in a [March 2024 discussion](https://github.com/orgs/LeelaChessZero/discussions/1981). That is a candidate source, not a verified manifest for the requested paper's 53-million-game April 2024 subset or its 500-million-game larger corpus. The archive could not be retrieved from this environment on 2026-10-01. No authentic chunk has yet passed this importer.

Example commands for **real chunks supplied separately**; these were not run on authentic self-play data here:

```sh
python3 -m chessformer.native_data --inputs /path/to/complete-games.tar \
  --out data/selfplay.sqlite --input-format 4 --validation-fraction 0.05

python3 -m chessformer.train --config configs/pilot.json \
  --train data/selfplay.sqlite --validation data/selfplay.sqlite \
  --out runs/native-pilot
```

The trainer selects the two named splits from the same database. All auxiliary labels are available; `--allow-missing-aux` is unnecessary. Start with a small validation run before using a paper-scale configuration.

[native_encoding.py](../chessformer/native_encoding.py) implements native inference, including file/rank/diagonal canonicalization, repetition-only history variants, castling/history cutoffs, EP flags, and the trainer's legacy rule-50 scaling (`/99` for formats 1–3, `/100` otherwise). It follows the engine's `FillEmptyHistory::NO` convention, including one inferable preceding pawn move for a legacy FEN with an explicit EP field. A chunk does not identify its producer's empty-history option; that setting must be checked against the eventual data source. Standard-chess legal actions match the pinned 1,858-index policy map.
