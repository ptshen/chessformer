"""Validated records and a bounded PGN bootstrap for local pipeline experiments."""

import argparse
import hashlib
import json
from pathlib import Path
import random

import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset

from .encoding import ENCODING, NUM_ACTIONS, board_from_record, encode_board, legal_moves_by_id, move_id


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint():
    """Hash numerical/data/training implementation for reproducible checkpoints."""
    root = Path(__file__).parent
    names = ("encoding.py", "model.py", "data.py", "losses.py", "train.py", "native.py", "native_data.py", "native_encoding.py", "native_validation.py")
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def record_arrays(record, require_aux=True):
    if record.get("encoding") != ENCODING:
        raise ValueError(f"record encoding must be {ENCODING}")
    if not isinstance(record.get("game_id"), str) or not record["game_id"]:
        raise ValueError("game_id is required to audit split leakage")
    board = board_from_record(record)
    if not board.is_valid() or board.is_game_over():
        raise ValueError("training position must be valid and nonterminal")
    legal = legal_moves_by_id(board)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[list(legal)] = True
    target = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for uci, weight in record["policy"].items():
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves or not np.isfinite(weight) or weight < 0:
            raise ValueError(f"invalid policy entry: {uci}: {weight}")
        target[move_id(move, board.turn)] = weight
    if target.sum() <= 0 or not np.isfinite(target.sum()):
        raise ValueError("policy must have finite positive mass")
    target /= target.sum()
    wdl = np.asarray(record["wdl"], dtype=np.float32)
    if wdl.shape != (3,) or not np.isfinite(wdl).all() or (wdl < 0).any() or not np.isclose(wdl.sum(), 1):
        raise ValueError("wdl must be normalized [win, draw, loss] in side-to-move perspective")
    result = {"inputs": encode_board(board), "legal": mask, "policy": target, "wdl": wdl}
    for key in ("q", "st"):
        valid = key in record and record[key] is not None
        if require_aux and not valid:
            raise ValueError(f"missing {key} target; PGN pilots require --allow-missing-aux")
        value = float(record[key]) if valid else 0.0
        if not np.isfinite(value) or not -1 <= value <= 1:
            raise ValueError(f"{key} must be finite in [-1,1]")
        result[key] = np.float32(value)
        result[key + "_valid"] = np.bool_(valid)
    return result


class PositionDataset(Dataset):
    """In-memory JSONL loader for pilots, not the full paper's billion-position data."""

    def __init__(self, path, require_aux=True):
        self.path = str(path)
        self.fingerprint = sha256_file(path)
        self.records = []
        self.samples = []
        with open(path) as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                    sample = record_arrays(record, require_aux)
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
                self.records.append(record)
                self.samples.append(sample)
        if not self.samples:
            raise ValueError(f"empty dataset: {path}")
        self.game_ids = {r["game_id"] for r in self.records}
        self.encoding = ENCODING
        self.sources = sorted({r.get("source", "unspecified") for r in self.records})
        self.available_aux = self.aux_counts(range(len(self)))

    @property
    def game_count(self):
        return len(self.game_ids)

    def aux_counts(self, indices):
        return {k: sum(bool(self.samples[int(i)][k + "_valid"]) for i in indices) for k in ("q", "st")}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {k: torch.as_tensor(v) for k, v in self.samples[index].items()}

    def batch(self, indices, device):
        return {k: torch.as_tensor(np.stack([self.samples[int(i)][k] for i in indices]), device=device)
                for k in self.samples[0]}


def assert_disjoint(train, validation):
    if train.encoding != validation.encoding:
        raise ValueError("train/validation encodings must match")
    if hasattr(train, "assert_disjoint"):
        return train.assert_disjoint(validation)
    overlap = train.game_ids & validation.game_ids
    if overlap:
        raise ValueError(f"train/validation overlap: {len(overlap)} game IDs")
    # Identical model inputs can occur in different games, especially openings.
    signatures = {hashlib.sha256(s["inputs"].tobytes()).digest() for s in train.samples}
    collisions = sum(hashlib.sha256(s["inputs"].tobytes()).digest() in signatures for s in validation.samples)
    if collisions:
        raise ValueError(f"train/validation contain {collisions} identical encoded positions")


def load_dataset(path, split, require_aux=True):
    if Path(path).suffix == ".sqlite":
        from .native_data import NativeDataset
        return NativeDataset(path, split)
    return PositionDataset(path, require_aux=require_aux)


def pgn_records(paths, max_games, positions_per_game, seed):
    rng = random.Random(seed)
    count = 0
    seen_games = set()
    for path in sorted(map(Path, paths)):
        with path.open() as stream:
            while count < max_games:
                game = chess.pgn.read_game(stream)
                if game is None:
                    break
                if game.errors or game.headers.get("Result") not in ("1-0", "0-1", "1/2-1/2"):
                    continue
                moves = [m.uci() for m in game.mainline_moves()]
                initial = game.board().fen()
                identity = json.dumps([initial, moves], separators=(",", ":"))
                game_id = hashlib.sha256(identity.encode()).hexdigest()
                if game_id in seen_games:
                    continue
                seen_games.add(game_id)
                count += 1
                # Skip early shared openings; remaining duplicates removed across splits below.
                candidates = list(range(16, len(moves)))
                chosen = set(rng.sample(candidates, min(positions_per_game, len(candidates))))
                board = game.board()
                winner = {"1-0": chess.WHITE, "0-1": chess.BLACK, "1/2-1/2": None}[game.headers["Result"]]
                for ply, uci in enumerate(moves):
                    if ply in chosen and not board.is_game_over():
                        value = 1 if winner is None else 0 if winner == board.turn else 2
                        yield {"encoding": ENCODING, "game_id": game_id, "source": "ccrl-pgn-bootstrap",
                               "initial_fen": initial, "moves": moves[:ply], "policy": {uci: 1.0},
                               "wdl": [int(j == value) for j in range(3)]}
                    board.push_uci(uci)
        if count >= max_games:
            break


def prepare_pgn(root, out, train_games=128, validation_games=32, positions_per_game=8, seed=17):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / name for name in ("train.jsonl", "validation.jsonl", "manifest.json")]
    if any(p.exists() for p in paths):
        raise FileExistsError("output dataset already exists; choose a new directory")
    fingerprints, game_ids = set(), set()
    summary = {"kind": "PGN smoke/pilot only: played-move policy, game result, no search values",
               "seed": seed, "encoding": ENCODING, "positions_per_game": positions_per_game,
               "requested_games": {"train": train_games, "validation": validation_games}}
    for split, count in (("train", train_games), ("validation", validation_games)):
        records = pgn_records((Path(root) / ("test" if split == "validation" else "train")).glob("*.pgn"),
                              count, positions_per_game, seed + int(split == "validation"))
        n, duplicates, own_games = 0, 0, set()
        path = out / f"{split}.jsonl"
        with path.open("w") as stream:
            for record in records:
                fingerprint = hashlib.sha256(encode_board(board_from_record(record)).tobytes()).digest()
                if fingerprint in fingerprints or (split == "validation" and record["game_id"] in game_ids):
                    duplicates += 1
                    continue
                fingerprints.add(fingerprint)
                own_games.add(record["game_id"])
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                n += 1
        game_ids.update(own_games)
        summary[split] = {"positions": n, "games": len(own_games), "duplicates_removed": duplicates,
                          "sha256": sha256_file(path)}
        if not n:
            raise ValueError(f"no usable {split} positions under {root}")
    (out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-games", type=int, default=128)
    parser.add_argument("--validation-games", type=int, default=32)
    parser.add_argument("--positions-per-game", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(prepare_pgn(args.pgn_root, args.out, args.train_games, args.validation_games,
                                 args.positions_per_game, args.seed), indent=2))
