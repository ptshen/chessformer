import random

import chess
import numpy as np
import pytest

from chessformer.encoding import encode_board, legal_moves_by_id, move_id


def test_history_uses_current_mover_perspective():
    board = chess.Board()
    board.push_uci("e2e4")
    x = encode_board(board)
    assert x[chess.E2, 0] == 1  # black e7 pawn is now own pawn at e2
    assert x[chess.E5, 6] == 1  # white e4 pawn is opposing pawn at e5
    assert x[chess.E7, 13 + 6] == 1  # old white e2 remains opposing pawn
    assert np.all(x[:, 26:104] == 0)  # no manufactured history
    assert x[chess.E8, 108] == 1


def test_color_mirror_invariance_including_moves():
    board = chess.Board("r3k2r/ppp2ppp/2n5/3pp3/4P3/5N2/PPPP1PPP/R3K2R w KQkq - 12 13")
    flipped = board.mirror()
    np.testing.assert_array_equal(encode_board(board), encode_board(flipped))
    ids = set(legal_moves_by_id(board))
    assert ids == set(legal_moves_by_id(flipped))


@pytest.mark.parametrize("fen", [
    "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
    "4k3/P7/8/8/8/8/7p/4K3 b - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
])
def test_special_moves_have_distinct_legal_actions(fen):
    board = chess.Board(fen)
    mapping = legal_moves_by_id(board)
    assert len(mapping) == board.legal_moves.count()
    assert set(mapping.values()) == set(board.legal_moves)


def test_all_legal_actions_in_random_games():
    rng = random.Random(314)
    for _ in range(10):
        board = chess.Board()
        for _ in range(150):
            moves = list(board.legal_moves)
            if board.is_game_over():
                break
            ids = [move_id(move, board.turn) for move in moves]
            assert len(set(ids)) == len(ids)
            assert all(0 <= i < 1858 for i in ids)
            board.push(rng.choice(moves))


def test_repetition_and_castling_rights():
    board = chess.Board()
    for move in "g1f3 g8f6 f3g1 f6g8".split():
        board.push_uci(move)
    x = encode_board(board)
    assert np.all(x[:, 12] == 1)
    assert x[[chess.A1, chess.A8], 104].sum() == 2
    assert x[[chess.H1, chess.H8], 105].sum() == 2
    assert x[0, 109] == np.float32(0.04)
