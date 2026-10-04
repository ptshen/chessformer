"""Synthetic protocol fixtures only; these are not Lc0 self-play experiments."""

import gzip
import io
import random
import struct
import tarfile

import chess
import numpy as np

from chessformer.native_encoding import canonical_transform, encode_native_board, native_legal_moves_by_id


def record(board, version=6, root_q=0.2, result_q=0., result_d=1., *, input_format=4, played_move=None):
    raw = bytearray(8356 if version == 6 else 8396)
    struct.pack_into("<II", raw, 0, version, input_format)
    legal = sorted(native_legal_moves_by_id(board, input_format))
    index = legal[0] if played_move is None else next(i for i, m in native_legal_moves_by_id(board, input_format).items() if m == played_move)
    struct.pack_into("<H", raw, 8344, index)
    policy = np.full(1858, -1., dtype="<f4")
    policy[legal] = 0.
    policy[legal[0]] = 0.7
    policy[legal[-1]] = 0.3
    raw[8:7440] = policy.tobytes()
    planes = encode_native_board(board, input_format)
    raw[7440:8272] = np.packbits(planes[:, :104].T.astype(bool)).tobytes()
    if input_format == 1:
        raw[8272:8276] = bytes(int(planes[0, c]) for c in range(104, 108))
    else:
        raw[8272:8276] = bytes(sum(int(planes[base + file, channel]) << file for file in range(8))
                              for base, channel in ((0, 104), (0, 105), (56, 104), (56, 105)))
    raw[8276] = (int(board.turn == chess.BLACK) if input_format < 3 else
                 sum(int(planes[56 + file, 108]) << file for file in range(8)))
    raw[8277] = board.halfmove_clock
    raw[8278] = canonical_transform(board, input_format) | (128 if board.turn == chess.BLACK else 0)
    struct.pack_into("<f", raw, 8280, root_q)
    struct.pack_into("<f", raw, 8288, 0.1)
    struct.pack_into("<ff", raw, 8308, result_q, result_d)
    struct.pack_into("<f", raw, 8352, 123.)  # V6 reserved/Q-std is intentionally NOT a reward.
    if version == 7:
        struct.pack_into("<ff", raw, 8352, -0.4, 0.2)
    return bytes(raw)


def game_bytes(seed=0, plies=8, version=6, input_format=4):
    rng = random.Random(seed)
    board = chess.Board()
    records = []
    for ply in range(plies):
        if board.is_game_over():
            break
        move = rng.choice(list(board.legal_moves))
        records.append(record(board, version=version, root_q=0.1 + (ply % 3) * 0.2,
                              input_format=input_format, played_move=move))
        board.push(move)
    return b"".join(records)


def write_archive(path, games=16):
    with tarfile.open(path, "w") as archive:
        for seed in range(games):
            payload = gzip.compress(game_bytes(seed), mtime=0)
            entry = tarfile.TarInfo(f"game-{seed}.gz")
            entry.size = len(payload)
            archive.addfile(entry, io.BytesIO(payload))
    return path
