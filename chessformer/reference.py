"""Fetch/check pinned public sources and run the authors' decoder as an oracle.

Downloaded GPL source retains its notices in data/upstream and is never imported
by production training. This verifier extracts only the format decoder/functions.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

import numpy as np


def git_blob_sha(content):
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()


def check_sources(manifest="results/upstream-sources.json", fetch=False):
    specification = json.loads(Path(manifest).read_text())
    for entry in specification["files"]:
        path = Path(entry["local_path"])
        if not path.exists() and fetch:
            url = f"https://raw.githubusercontent.com/{entry['repository']}/{entry['commit']}/{entry['path']}"
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read()
            if git_blob_sha(content) != entry["blob_sha"]:
                raise ValueError(f"source hash mismatch: {url}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        if not path.exists():
            raise FileNotFoundError(f"{path}; fetch pinned sources with python3 -m chessformer.reference --fetch")
        if git_blob_sha(path.read_bytes()) != entry["blob_sha"]:
            raise ValueError(f"source hash mismatch: {path}")
    return specification


def decoder_oracle(specification=None):
    specification = check_sources() if specification is None else specification
    entry = next(e for e in specification["files"] if e["path"] == "tf/chunkparser.py")
    # Importing the whole upstream module would also import multiprocessing code.
    # Compile only the selected constants and pure functions after SHA verification.
    tree = ast.parse(Path(entry["local_path"]).read_text())
    constants = {"n_future_probs", "n_future_boards", "V6_VERSION", "V7_VERSION", "V7B_VERSION",
                 "V7B_STRUCT_STRING", "v7b_struct"}
    functions = {"reverse_expand_bits", "convert_v7b_to_tuple", "apply_alpha"}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in constants:
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
    namespace = {"np": np, "struct": struct}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), entry["path"], "exec"), namespace)
    return namespace


def verify_native(seed=29, cases_per_format=8):
    from .encoding import ACTIONS
    from .native import INPUT_FORMATS, decode_record, short_term_values
    import chess

    specification = check_sources()
    oracle = decoder_oracle(specification)
    index_source = next(e for e in specification["files"] if e["path"] == "tf/policy_index.py")
    expected_indices = ast.literal_eval(ast.parse(Path(index_source["local_path"]).read_text()).body[0].value)
    actual_indices = [chess.square_name(a) + chess.square_name(b) + (chess.piece_symbol(p) if p else "") for a, b, p in ACTIONS]
    if actual_indices != expected_indices:
        raise AssertionError("native policy ordering differs from authors' policy_index.py")
    rng = np.random.default_rng(seed)
    errors = {"inputs": 0., "policy": 0., "wdl": 0., "q": 0., "st": 0.}
    cases = 0
    for input_format in INPUT_FORMATS:
        for _ in range(cases_per_format):
            # Generated protocol fixtures exercise all planes and byte flags; these
            # are not authentic games or evidence of chess strength.
            raw = bytearray(8396)
            struct.pack_into("<II", raw, 0, 7, input_format)
            policy = np.full(1858, -1., dtype="<f4")
            ids = rng.choice(1858, 30, replace=False)
            masses = rng.random(30).astype(np.float32)
            masses[0] = 0  # legal but never visited
            policy[ids] = masses / masses.sum()
            raw[8:7440] = policy.tobytes()
            raw[7440:8272] = rng.integers(0, 256, 832, dtype=np.uint8).tobytes()
            if input_format == 1:
                raw[8272:8277] = rng.integers(0, 2, 5, dtype=np.uint8).tobytes()
            else:
                raw[8272:8276] = rng.integers(0, 256, 4, dtype=np.uint8).tobytes()
                raw[8276] = int(rng.integers(0, 2 if input_format == 2 else 256))
            raw[8277] = int(rng.integers(0, 100))
            raw[8278] = int(rng.choice([0, 128]))
            q, st = rng.uniform(-0.7, 0.7, 2)
            struct.pack_into("<f", raw, 8280, q)
            struct.pack_into("<f", raw, 8288, 0.1)
            struct.pack_into("<ff", raw, 8308, 0.25, 0.25)
            struct.pack_into("<ff", raw, 8352, st, 0.2)
            actual = decode_record(raw)
            converted = oracle["convert_v7b_to_tuple"](raw + bytes(oracle["v7b_struct"].size - len(raw)))
            expected = {"inputs": np.frombuffer(converted[0], dtype=np.float32).reshape(112, 64).T,
                        "policy": np.maximum(np.frombuffer(converted[1], dtype=np.float32), 0),
                        "wdl": np.frombuffer(converted[2], dtype=np.float32)}
            expected["policy"] /= expected["policy"].sum()
            wdl_q, wdl_st = np.frombuffer(converted[3], dtype=np.float32), np.frombuffer(converted[5], dtype=np.float32)
            expected["q"] = wdl_q[0] - wdl_q[2]
            expected["st"] = wdl_st[0] - wdl_st[2]
            for key in expected:
                difference = float(np.abs(actual[key] - expected[key]).max())
                errors[key] = max(errors[key], difference)
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6, atol=1e-7)
            cases += 1
    ema_error = 0.
    for n in (1, 2, 8, 100, 1000):
        rewards = rng.uniform(-1, 1, n).astype(np.float32)
        expected = oracle["apply_alpha"](rewards, 5 / 6)
        actual = short_term_values(rewards)
        ema_error = max(ema_error, float(np.abs(actual - expected).max()))
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
    return {"scope": "synthetic binary conformance; no authentic self-play data tested",
            "training_commit": index_source["commit"], "seed": seed, "cases": cases,
            "input_formats": list(INPUT_FORMATS), "policy_actions_matching": len(ACTIONS),
            "maximum_absolute_errors": errors, "ema_maximum_absolute_error": ema_error,
            "source_files_verified": len(specification["files"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--verify-native", action="store_true")
    parser.add_argument("--out", default="results/native-conformance.json")
    args = parser.parse_args()
    specification = check_sources(fetch=args.fetch)
    if args.verify_native:
        report = verify_native()
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    else:
        print(f"Verified {len(specification['files'])} pinned source files")
