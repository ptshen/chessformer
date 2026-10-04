"""Compile pinned Lc0 sources and compare native input planes and move indices.

Requires a C++20 compiler, protoc and protobuf headers. No neural
backend, external engine process, search labels or trained weights are used.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import struct
import subprocess

import chess
import numpy as np

from .native import INPUT_FORMATS
from .native_encoding import canonical_transform, encode_native_board, native_legal_moves_by_id
from .native_validation import current_board, position_key
from .reference import check_sources


def build_oracle(out, include_dirs=()):
    specification = check_sources()
    entry = next(e for e in specification["files"] if e["path"] == "src/neural/encoder.cc"
                 and e["commit"] == specification["engine_reference_commit"])
    proto = next(e for e in specification["files"] if e["repository"] == "LeelaChessZero/lczero-common"
                 and e["path"] == "proto/net.proto")
    proto_root = Path(proto["local_path"]).parents[1]
    root = Path(entry["local_path"]).parents[2]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    includes = ["-I", str(root / "src"), "-I", str(out)]
    for directory in include_dirs:
        includes += ["-I", str(directory)]
    flags = ["-std=c++20", "-O2", "-DNO_PEXT"] + includes
    subprocess.run(["protoc", f"--cpp_out={out}", "-I", str(proto_root),
                    str(proto_root / "proto/net.proto")], check=True)
    subprocess.run(compiler + flags + ["-c", str(root / "src/utils/logging.cc"),
                   "-o", str(out / "logging.o")], check=True)
    executable = out / "encoder"
    subprocess.run(compiler + flags + ["tools/native_encoder_oracle.cc",
                   *[str(root / path) for path in ("src/chess/board.cc", "src/chess/bitboard.cc", "src/chess/position.cc",
                                                  "src/neural/encoder.cc")],
                   str(out / "logging.o"), "-o", str(executable)], check=True)
    return executable, entry["commit"], subprocess.check_output(compiler + ["--version"], text=True).splitlines()[0]


def comparison_boards(seed=47):
    rng = random.Random(seed)
    cases = []
    sequences = ["", "e2e4", "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6",
                 "e2e4 g7g6 e4e5 f7f5 g1f3 g8f6",
                 "e2e4 e7e5 e1e2 e8e7 e2d3 e7e6"]
    for sequence in sequences:
        board = chess.Board()
        cases.append(board.copy())
        for uci in sequence.split():
            board.push_uci(uci)
            cases.append(board.copy())
    fens = ["3r4/4k3/8/1K6/8/8/8/8 w - - 0 1",
            "3r4/4k3/2K5/8/8/8/8/8 w - - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 17 1",
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 17 1",
            "7k/P7/8/8/8/8/8/K7 w - - 0 1",
            "k7/8/8/8/8/8/p7/7K b - - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "k3r3/8/8/4Pp2/8/8/8/4K3 w - f6 0 1"]
    for fen in fens:
        cases.append(chess.Board(fen))
    # Pawnless asymmetric positions exercise all canonical transforms and ties.
    for king in range(64):
        for piece in (chess.QUEEN, chess.ROOK, chess.KNIGHT, chess.BISHOP):
            board = chess.Board(None)
            squares = rng.sample([s for s in range(64) if s != king], 3)
            for square, kind, color in [(king, chess.KING, chess.WHITE),
                                        (squares[0], chess.KING, chess.BLACK),
                                        (squares[1], piece, chess.WHITE),
                                        (squares[2], chess.KNIGHT, chess.BLACK)]:
                board.set_piece_at(square, chess.Piece(kind, color))
            if board.is_valid():
                cases.append(board)
    # Legal move histories, including captures, rights changes and late games.
    for _ in range(12):
        board = chess.Board()
        for ply in range(160):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
            if ply % 7 == 0:
                cases.append(board.copy())
    return cases


def verify_encoder(executable, commit, compiler, seed=47):
    boards = comparison_boards(seed)
    requests, cases = [], []
    for board in boards:
        for fmt in INPUT_FORMATS:
            requests.append(f"{fmt}\t{board.root().fen(en_passant='fen')}\t" +
                            " ".join(move.uci() for move in board.move_stack))
            cases.append((board, fmt))
    payload = "\n".join(requests) + "\n"
    result = subprocess.run([str(executable)], input=payload, text=True, capture_output=True, check=True)
    lines = result.stdout.splitlines()
    if len(lines) != len(cases):
        raise AssertionError("C++ oracle returned a different number of cases")
    error, legal_moves, transforms = 0., 0, set()
    for (board, fmt), line in zip(cases, lines):
        expected = json.loads(line)
        transform = canonical_transform(board, fmt)
        if transform != expected["transform"]:
            raise AssertionError(f"canonical transform mismatch for {board.fen()}, format {fmt}")
        transforms.add(transform)
        planes = np.zeros((64, 112), dtype=np.float32)
        for channel, (mask, value) in enumerate(expected["planes"]):
            for square in chess.scan_forward(mask):
                planes[square, channel] = value
        # The native trainer scales legacy raw rule50 planes by 99.
        if fmt < 4:
            planes[:, 109] /= 99
        actual = encode_native_board(board, fmt)
        np.testing.assert_allclose(actual, planes, atol=1e-7, rtol=0,
                                   err_msg=f"{board.fen()}, format {fmt}")
        error = max(error, float(np.abs(actual - planes).max()))
        moves = {index: move.uci() for index, move in native_legal_moves_by_id(board, fmt).items()}
        expected_moves = {index: board.parse_uci(uci).uci() for index, uci in expected["legal"]}
        if moves != expected_moves:
            raise AssertionError(f"legal action mismatch for {board.fen()}, format {fmt}")
        # Pack the compiled encoder's planes in the pinned writer's V6 layout.
        # This is a synthetic bridge, not a real self-play chunk or search label.
        raw = bytearray(8356)
        struct.pack_into("<II", raw, 0, 6, fmt)
        raw[7440:8272] = np.packbits(planes[:, :104].T.astype(bool)).tobytes()
        if fmt == 1:
            raw[8272:8276] = bytes(int(planes[0, c]) for c in range(104, 108))
        else:
            raw[8272:8276] = bytes(sum(int(planes[base + file, channel]) << file for file in range(8))
                                  for base, channel in ((0, 104), (0, 105), (56, 104), (56, 105)))
        raw[8276] = (int(board.turn == chess.BLACK) if fmt < 3 else
                     sum(int(planes[56 + file, 108]) << file for file in range(8)))
        raw[8277] = board.halfmove_clock
        raw[8278] = (transform | (128 if board.turn == chess.BLACK else 0)) if fmt >= 3 else 0
        decoded = current_board(bytes(raw))
        if position_key(decoded, False) != position_key(board, False):
            raise AssertionError(f"native current-position decoding mismatch for {board.fen()}, format {fmt}")
        decoded_moves = {index: move.uci() for index, move in native_legal_moves_by_id(decoded, fmt).items()}
        if decoded_moves != expected_moves:
            raise AssertionError(f"decoded legal moves mismatch for {board.fen()}, format {fmt}")
        if fmt >= 3 and position_key(decoded) != position_key(board):
            raise AssertionError(f"decoded canonical EP mismatch for {board.fen()}, format {fmt}")
        legal_moves += len(moves)
    return {"scope": "native inference comparison with compiled, unmodified pinned Lc0 chess/encoder sources; no training data",
            "lc0_commit": commit, "compiler": compiler, "empty_history": "NO",
            "seed": seed, "board_histories": len(boards), "format_cases": len(cases),
            "input_formats": list(INPUT_FORMATS), "transforms_observed": sorted(transforms),
            "legal_move_indices_matched": legal_moves, "maximum_input_absolute_error": error,
            "native_current_positions_decoded": len(cases), "decoded_legal_move_indices_matched": legal_moves,
            "native_decode_scope": "current planes from compiled Lc0 encoded in synthetic V6 frames; no search labels or authentic chunks",
            "request_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "oracle_sha256": hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
            "driver_sha256": hashlib.sha256(Path("tools/native_encoder_oracle.cc").read_bytes()).hexdigest(),
            "python_encoder_sha256": hashlib.sha256(Path("chessformer/native_encoding.py").read_bytes()).hexdigest(),
            "python_native_validation_sha256": hashlib.sha256(Path("chessformer/native_validation.py").read_bytes()).hexdigest(),
            "python_reference_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="runs/native-oracle")
    parser.add_argument("--include-dir", action="append", default=[])
    parser.add_argument("--out", default="results/native-encoder-conformance.json")
    args = parser.parse_args()
    executable, commit, compiler = build_oracle(args.build_dir, args.include_dir)
    report = verify_encoder(executable, commit, compiler)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
