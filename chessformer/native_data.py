"""Disk-backed native self-play dataset; preprocessing memory is bounded per game."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import torch

from .data import sha256_file
from .native import iter_games, native_encoding

SCHEMA = "chessformer-native-sqlite-v2"
FIELDS = "bits,rule50,moves,policy,wdl,q,st"


def _connect(path, readonly=False):
    uri = Path(path).resolve().as_uri() + "?mode=ro" if readonly else str(path)
    db = sqlite3.connect(uri, uri=readonly)
    db.execute("PRAGMA cache_size=-32768")  # bounded 32 MiB page cache
    db.execute("PRAGMA foreign_keys=ON")
    return db


def game_split(game_id, seed, validation_fraction):
    value = int.from_bytes(hashlib.sha256(f"{seed}:{game_id}".encode()).digest()[:8], "big")
    return "validation" if value / 2**64 < validation_fraction else "train"


def build_native_dataset(inputs, out, *, input_format, validation_fraction=0.05, seed=17,
                         sample_stride=1, max_plies=8192, max_games=None):
    """A finalized database is atomic; a failed import never replaces a good dataset."""
    if not 0 < validation_fraction < 1 or sample_stride < 1 or (max_games is not None and max_games < 1):
        raise ValueError("invalid split fraction, sampling stride, or game limit")
    encoding = native_encoding(input_format)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".building")
    if out.exists() or temporary.exists():
        raise FileExistsError("dataset or unfinished import exists; choose a new output path")
    db = _connect(temporary)
    manifest = {"schema": SCHEMA, "encoding": encoding, "input_format": input_format,
                "sources": ["lc0-native-chunks"], "inputs": [str(Path(p)) for p in inputs],
                "validation_fraction": validation_fraction, "seed": seed, "sample_stride": sample_stride,
                "max_plies": max_plies, "max_games": max_games, "games_read": 0,
                "chronology_validation": "standard-chess played_idx replay, current planes, rights, EP, and rule50",
                "chunk_endpoints_verified": False,
                "duplicate_games": 0, "duplicate_inputs": 0, "deleted_frames": 0,
                "versions": {}, "counts": {"train": 0, "validation": 0}}
    try:
        db.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE games (game_id TEXT PRIMARY KEY, split TEXT NOT NULL,
                source TEXT NOT NULL, content_sha256 TEXT NOT NULL, plies INTEGER NOT NULL);
            CREATE TABLE positions (split TEXT NOT NULL, id INTEGER NOT NULL,
                game_id TEXT NOT NULL REFERENCES games(game_id), ply INTEGER NOT NULL,
                input_sha256 BLOB NOT NULL UNIQUE, bits BLOB NOT NULL, rule50 REAL NOT NULL,
                moves BLOB NOT NULL, policy BLOB NOT NULL, wdl BLOB NOT NULL,
                q REAL NOT NULL, st REAL NOT NULL, PRIMARY KEY(split,id));
            CREATE INDEX positions_game ON positions(game_id);
        """)
        for game in iter_games(inputs, max_plies=max_plies, expected_input_format=input_format):
            manifest["games_read"] += 1
            version = str(game.version)
            manifest["versions"][version] = manifest["versions"].get(version, 0) + 1
            split = game_split(game.game_id, seed, validation_fraction)
            with db:
                inserted = db.execute("INSERT OR IGNORE INTO games VALUES (?,?,?,?,?)",
                                      (game.game_id, split, game.source, game.content_sha256, len(game.records))).rowcount
                if not inserted:
                    manifest["duplicate_games"] += 1
                else:
                    manifest["deleted_frames"] += sum(bool(raw[8278] & 64) for raw in game.records)
                    offset = int(game.game_id[:8], 16) % sample_stride
                    for ply, sample in game.samples():
                        if ply % sample_stride != offset:
                            continue
                        x = sample["inputs"]
                        digest = hashlib.sha256(x.tobytes()).digest()
                        legal = np.flatnonzero(sample["legal"]).astype("<u2")
                        values = (split, manifest["counts"][split], game.game_id, ply, digest,
                                  np.packbits(x.T.astype(bool)).tobytes(), float(x[0, 109]), legal.tobytes(),
                                  sample["policy"][legal].astype("<f4").tobytes(), sample["wdl"].astype("<f4").tobytes(),
                                  float(sample["q"]), float(sample["st"]))
                        inserted = db.execute("INSERT OR IGNORE INTO positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", values).rowcount
                        if inserted:
                            manifest["counts"][split] += 1
                        else:
                            manifest["duplicate_inputs"] += 1
            if max_games is not None and manifest["games_read"] >= max_games:
                break
        if not all(manifest["counts"].values()):
            raise ValueError("import needs nonempty train and validation splits; provide more games or change split seed")
        manifest["games"] = {split: db.execute("SELECT COUNT(DISTINCT game_id) FROM positions WHERE split=?", (split,)).fetchone()[0]
                             for split in manifest["counts"]}
        # The database itself is the authoritative manifest; sidecar is for inspection.
        with db:
            db.execute("INSERT INTO metadata VALUES ('manifest', ?)", (json.dumps(manifest, sort_keys=True),))
        db.close()
        os.replace(temporary, out)
        manifest["database_sha256"] = sha256_file(out)
        out.with_suffix(out.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    except BaseException:
        db.close()
        # Keep the partial database for inspection, but it cannot be opened as finalized.
        raise


class NativeDataset:
    """Indexed sparse policies and packed board bits, without loading the corpus into RAM."""

    def __init__(self, path, split):
        if split not in ("train", "validation"):
            raise ValueError("split must be train or validation")
        self.path, self.split = str(Path(path).resolve()), split
        self.db = _connect(self.path, readonly=True)
        try:
            row = self.db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
            if row is None:
                raise ValueError("unfinished native import")
            self.manifest = json.loads(row[0])
            if self.manifest["schema"] != SCHEMA:
                raise ValueError("unsupported native dataset schema; reimport complete chunks to validate chronology")
            self.encoding = self.manifest["encoding"]
            if self.encoding != native_encoding(self.manifest["input_format"]):
                raise ValueError("native encoding and input format disagree")
            self.n = self.manifest["counts"][split]
            if self.n <= 0:
                raise ValueError("empty native dataset split")
            self.game_count = self.manifest["games"][split]
            self.sources = self.manifest["sources"]
            self.available_aux = {"q": self.n, "st": self.n}
            self.database_sha256 = sha256_file(self.path)
            self.fingerprint = hashlib.sha256(f"{self.database_sha256}:{split}".encode()).hexdigest()
        except BaseException:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def __len__(self):
        return self.n

    def aux_counts(self, indices):
        return {"q": len(indices), "st": len(indices)}

    def batch(self, indices, device):
        samples = []
        for index in indices:
            index = int(index)
            if not 0 <= index < self.n:
                raise IndexError(index)
            row = self.db.execute(f"SELECT {FIELDS} FROM positions WHERE split=? AND id=?", (self.split, index)).fetchone()
            if row is None:
                raise ValueError(f"missing native row {index}")
            bits, clock, ids, probabilities, wdl, q, st = row
            x = np.unpackbits(np.frombuffer(bits, dtype=np.uint8)).reshape(112, 64).T.astype(np.float32)
            x[:, 109] = clock
            moves = np.frombuffer(ids, dtype="<u2")
            legal = np.zeros(1858, dtype=bool)
            policy = np.zeros(1858, dtype=np.float32)
            legal[moves] = True
            policy[moves] = np.frombuffer(probabilities, dtype="<f4")
            samples.append({"inputs": x, "legal": legal, "policy": policy,
                            "wdl": np.frombuffer(wdl, dtype="<f4"), "q": np.float32(q), "st": np.float32(st),
                            "q_valid": np.bool_(True), "st_valid": np.bool_(True)})
        if not samples:
            raise ValueError("empty batch")
        return {key: torch.as_tensor(np.stack([s[key] for s in samples]), device=device) for key in samples[0]}

    def assert_disjoint(self, other):
        if not isinstance(other, NativeDataset) or self.encoding != other.encoding:
            raise ValueError("train/validation native encodings must match")
        # SQLite indexes perform overlap checks on disk rather than materializing
        # millions of game IDs or feature hashes in Python sets.
        if self.path == other.path:
            alias, attached = "main", False
        else:
            uri = Path(other.path).as_uri() + "?mode=ro"
            self.db.execute("ATTACH DATABASE ? AS validation_db", (uri,))
            alias, attached = "validation_db", True
        try:
            query = f"SELECT 1 FROM positions a JOIN {alias}.positions b ON a.input_sha256=b.input_sha256 WHERE a.split=? AND b.split=? LIMIT 1"
            if self.db.execute(query, (self.split, other.split)).fetchone():
                raise ValueError("train/validation contain identical encoded positions")
            query = f"SELECT 1 FROM games a JOIN {alias}.games b ON a.game_id=b.game_id WHERE a.split=? AND b.split=? LIMIT 1"
            if self.db.execute(query, (self.split, other.split)).fetchone():
                raise ValueError("train/validation overlap in native game IDs")
        finally:
            if attached:
                self.db.execute("DETACH DATABASE validation_db")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Complete-game .gz files or .tar archives")
    parser.add_argument("--out", required=True)
    parser.add_argument("--input-format", type=int, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=8192)
    parser.add_argument("--max-games", type=int)
    args = parser.parse_args()
    print(json.dumps(build_native_dataset(args.inputs, args.out, input_format=args.input_format,
                                         validation_fraction=args.validation_fraction, seed=args.seed,
                                         sample_stride=args.sample_stride, max_plies=args.max_plies,
                                         max_games=args.max_games), indent=2))
