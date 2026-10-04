import io
import struct

import chess
import pytest

from chessformer.native import INPUT_FORMATS, read_game
from chessformer.native_encoding import native_legal_moves_by_id
from chessformer.native_validation import current_board, position_key
from native_fixture import game_bytes, record


@pytest.mark.parametrize("fmt", INPUT_FORMATS)
@pytest.mark.parametrize("version", (6, 7))
def test_replay_rejects_two_missing_plies_despite_alternating_movers(fmt, version):
    size = 8356 if version == 6 else 8396
    raw = game_bytes(seed=14, plies=20, version=version, input_format=fmt)
    assert len(read_game(io.BytesIO(raw)).records) == 20
    with pytest.raises(ValueError, match="nonconsecutive"):
        read_game(io.BytesIO(raw[:size] + raw[3 * size:]))


@pytest.mark.parametrize("fmt", INPUT_FORMATS)
def test_replay_castling_ep_promotion_and_deleted_frame(fmt):
    cases = [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 17 1", ["e1g1", "e8c8", "f1f7"]),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ["e5d6", "e8f7", "d6d7"]),
        ("4k3/8/8/8/4Pp2/8/8/4K3 b - e3 0 1", ["f4e3", "e1f1", "e3e2"]),
        ("7k/P7/8/8/8/8/8/K7 w - - 0 1", ["a7a8n", "h8g8", "a1b2"]),
        ("k7/8/8/8/8/8/p7/7K b - - 0 1", ["a2a1q", "h1h2", "a1a2"]),
    ]
    for fen, moves in cases:
        board, frames = chess.Board(fen), []
        for uci in moves:
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves
            raw = record(board, input_format=fmt, played_move=move)
            decoded = current_board(raw)
            assert position_key(decoded, False) == position_key(board, False)
            assert native_legal_moves_by_id(decoded, fmt) == native_legal_moves_by_id(board, fmt)
            frames.append(raw)
            board.push(move)
        deleted = bytearray(frames[1])
        deleted[8278] |= 64
        frames[1] = bytes(deleted)
        game = read_game(io.BytesIO(b"".join(frames)))
        assert len(game.records) == 3 and [ply for ply, _ in game.samples()] == [0, 2]


def test_replay_rejects_wrong_recorded_move_clock_and_piece_overlap():
    original = game_bytes(seed=19)
    raw = bytearray(original)
    struct.pack_into("<H", raw, 8344, 1858)
    with pytest.raises(ValueError, match="played_idx.*not legal"):
        read_game(io.BytesIO(raw))
    board = current_board(original[:8356])
    legal = native_legal_moves_by_id(board, 4)
    index = struct.unpack_from("<H", original, 8344)[0]
    wrong = bytearray(original)
    struct.pack_into("<H", wrong, 8344, next(i for i in legal if i != index))
    with pytest.raises(ValueError, match="nonconsecutive"):
        read_game(io.BytesIO(wrong))
    wrong = bytearray(original)
    wrong[8356 + 8277] = 97
    with pytest.raises(ValueError, match="nonconsecutive"):
        read_game(io.BytesIO(wrong))
    wrong = bytearray(original)
    wrong[7448:7456] = wrong[7440:7448]
    with pytest.raises(ValueError, match="planes overlap"):
        read_game(io.BytesIO(wrong))


def test_replay_does_not_claim_to_detect_missing_chunk_endpoints():
    raw = game_bytes(seed=29, plies=20)
    # A legal contiguous subline cannot establish where self-play began/ended.
    assert len(read_game(io.BytesIO(raw[3 * 8356:12 * 8356])).records) == 9
