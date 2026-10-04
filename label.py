"""Stockfish action-value labeling on Modal.

Stage "boards": parse CCRL PGNs from the volume, sample positions per game,
dedupe globally, split into shards.
Stage "label": 64 containers each run Stockfish over one shard, scoring every
legal move per board at MOVE_TIME seconds per (state, action) pair.

Usage:
    modal run label.py --stage boards
    modal run label.py --stage label [--split train|test]
"""

import io
import json
import os
import random

import modal

NUM_SHARDS = 64
TRAIN_GAMES = 40_000
TEST_GAMES = 1_300  # ~45 boards/game -> >50k eval boards
# ~30.9 legal moves/board measured, so 60 boards/game over 40k games (less ~20%
# dedup) lands near 60M action-values. Test stays at 48: its labels are already
# computed and resampling would invalidate them.
TRAIN_BOARDS_PER_GAME = 60
TEST_BOARDS_PER_GAME = 48
MOVE_TIME = 0.02  # seconds of Stockfish per (state, action)
CHECKPOINT_EVERY = 2000  # boards

app = modal.App("chessformer-label")
vol = modal.Volume.from_name("cclr-data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("stockfish")
    .pip_install("chess==1.11.2", "numpy")
    .add_local_python_source("common")
)


@app.function(image=image, volumes={"/data": vol}, timeout=2 * 3600, memory=4096)
def build_boards(split: str, num_games: int, boards_per_game: int) -> None:
    import chess.pgn

    rng = random.Random(1234 if split == "train" else 4321)
    pgn_dir = f"/data/cclr/{'train' if split == 'train' else 'test'}"
    files = sorted(os.listdir(pgn_dir), key=lambda f: int(f.split(".")[0]))

    seen: set[str] = set()
    boards: list[str] = []
    games_read = 0
    for fname in files:
        if games_read >= num_games:
            break
        with open(os.path.join(pgn_dir, fname)) as fh:
            while games_read < num_games:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                games_read += 1
                board = game.board()
                fens = []
                for move in game.mainline_moves():
                    board.push(move)
                    if not board.is_game_over() and board.legal_moves.count() > 1:
                        fens.append(board.fen())
                if len(fens) > boards_per_game:
                    fens = rng.sample(fens, boards_per_game)
                for fen in fens:
                    key = " ".join(fen.split(" ")[:4])  # ignore move counters
                    if key not in seen:
                        seen.add(key)
                        boards.append(fen)
        print(f"{fname}: {games_read} games, {len(boards)} boards", flush=True)

    rng.shuffle(boards)
    os.makedirs("/data/boards", exist_ok=True)
    for i in range(NUM_SHARDS):
        with open(f"/data/boards/{split}_{i:02d}.json", "w") as f:
            json.dump(boards[i::NUM_SHARDS], f)
    vol.commit()
    print(f"{split}: {games_read} games -> {len(boards)} boards in {NUM_SHARDS} shards")


@app.function(
    image=image, volumes={"/data": vol}, timeout=12 * 3600, cpu=1.5, memory=2048, retries=2
)
def label_shard(split: str, shard: int) -> int:
    import chess
    import chess.engine
    import numpy as np

    import common

    with open(f"/data/boards/{split}_{shard:02d}.json") as f:
        fens = json.load(f)

    out_path = f"/data/labels/{split}_{shard:02d}.npz"
    os.makedirs("/data/labels", exist_ok=True)

    # resume from a previous partial checkpoint if present
    done = 0
    tokens, counts, move_ids, wins = [], [], [], []
    if os.path.exists(out_path):
        prev = np.load(out_path)
        done = int(prev["done"])
        # A checkpoint written against a different board list would silently
        # misalign every value with the wrong board. Refuse to resume it.
        if int(prev["n_fens"]) != len(fens) or (done and str(prev["fens"][0]) != fens[0]):
            raise RuntimeError(
                f"shard {shard}: checkpoint has {int(prev['n_fens'])} boards but the "
                f"current shard has {len(fens)}; delete /data/labels/{split}_{shard:02d}.npz"
            )
        tokens = list(prev["tokens"])
        counts = list(prev["counts"])
        move_ids = list(prev["move_ids"])
        wins = list(prev["wins"])
        if done >= len(fens):
            print(f"shard {shard} already complete")
            return done

    import shutil

    # Debian installs the binary to /usr/games, which is not on PATH.
    engine_path = shutil.which("stockfish") or "/usr/games/stockfish"
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"stockfish not found at {engine_path}")

    def start_engine():
        # 5s ceiling per command: a 20ms search that takes longer is wedged.
        eng = chess.engine.SimpleEngine.popen_uci(engine_path, timeout=5.0)
        eng.configure({"Threads": 1, "Hash": 32})
        return eng

    def save():
        np.savez_compressed(
            out_path + ".tmp",
            done=done,
            n_fens=len(fens),
            tokens=np.array(tokens, dtype=np.uint8),
            counts=np.array(counts, dtype=np.uint16),
            move_ids=np.array(move_ids, dtype=np.uint16),
            wins=np.array(wins, dtype=np.float16),
            fens=np.array(fens[:done]),
        )
        os.replace(out_path + ".tmp.npz", out_path)
        vol.commit()

    engine = start_engine()
    limit = chess.engine.Limit(time=MOVE_TIME)
    for fen in fens[done:]:
        board = chess.Board(fen)
        legal = sorted(board.legal_moves, key=lambda m: m.uci())
        board_wins = []
        for move in legal:
            for attempt in range(3):
                try:
                    info = engine.analyse(board, limit, root_moves=[move])
                    score = info["score"].relative.score(mate_score=32000)
                    if score is None:
                        raise ValueError("no score")
                    board_wins.append(common.cp_to_win_prob(score))
                    break
                # Catch broadly: a wedged engine surfaces as TimeoutError from
                # python-chess's internal wait, which a narrow tuple misses and
                # which then hangs the shard indefinitely.
                except Exception:
                    try:
                        engine.quit()
                    except Exception:
                        pass
                    engine = start_engine()
            else:
                board_wins.append(0.5)  # engine kept failing; neutral label
        tokens.append(common.tokenize_fen(fen))
        counts.append(len(legal))
        move_ids.extend(common.MOVE_TO_ID[m.uci()] for m in legal)
        wins.extend(board_wins)
        done += 1
        if done % CHECKPOINT_EVERY == 0:
            save()
            print(f"shard {shard}: {done}/{len(fens)} boards", flush=True)

    save()
    engine.quit()
    print(f"shard {shard} finished: {done} boards, {len(wins)} action-values")
    return len(wins)


@app.local_entrypoint()
def main(stage: str, split: str = "train", shards: str = ""):
    if stage == "boards":
        specs = {
            "train": ("train", TRAIN_GAMES, TRAIN_BOARDS_PER_GAME),
            "test": ("test", TEST_GAMES, TEST_BOARDS_PER_GAME),
        }
        chosen = [specs[split]] if split in specs else list(specs.values())
        for args in chosen:
            build_boards.spawn(*args)
        print(f"spawned board extraction for {[c[0] for c in chosen]}")
    elif stage == "label":
        # spawn, not map: .map()/.remote() are cancelled if the local client
        # disconnects, which kills a multi-hour run. Spawned calls are detached.
        idx = [int(s) for s in shards.split(",")] if shards else range(NUM_SHARDS)
        calls = [label_shard.spawn(split, i) for i in idx]
        print(f"spawned {len(calls)} shards for {split}; poll /data/labels for progress")
    else:
        raise ValueError(stage)
