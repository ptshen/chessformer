import chess
import numpy as np
import pytest
import torch

from chessformer.encoding import ACTION_TO_ID
from chessformer.native_encoding import (canonical_transform, encode_native_board,
                                        native_ep_square, native_legal_moves_by_id)
from chessformer.play import choose_move, encoding_functions


def position(*moves, fen=chess.STARTING_FEN):
    board = chess.Board(fen)
    for move in moves:
        board.push_uci(move)
    return board


@pytest.mark.parametrize("fen,transform,own_king,their_king", [
    ("3r4/4k3/8/1K6/8/8/8/8 w - - 0 1", 7, 12, 38),
    ("3r4/4k3/2K5/8/8/8/8/8 w - - 0 1", 3, 21, 11),
])
def test_pinned_lc0_canonical_examples(fen, transform, own_king, their_king):
    # Independent expectations from the pinned C++ encoder_test.cc.
    board = chess.Board(fen)
    assert canonical_transform(board) == transform
    planes = encode_native_board(board)
    assert np.flatnonzero(planes[:, 5]).tolist() == [own_king]
    assert np.flatnonzero(planes[:, 11]).tolist() == [their_king]


def test_native_start_and_black_castling_plane_variants():
    board = chess.Board()
    legacy = encode_native_board(board, 1)
    assert np.all(legacy[:, 104:108] == 1)
    assert not legacy[:, 13:104].any()
    for fmt in (2, 3, 4, 5, 132, 133):
        planes = encode_native_board(board, fmt)
        assert np.flatnonzero(planes[:, 104]).tolist() == [0, 56]
        assert np.flatnonzero(planes[:, 105]).tolist() == [7, 63]
        assert not planes[:, 106:111].any()
    board.push_uci("g1f3")
    assert np.all(encode_native_board(board, 1)[:, 108] == 1)
    assert np.all(encode_native_board(board, 132)[:, 110] == 1)
    assert encode_native_board(board, 3)[0, 109] == np.float32(1 / 99)
    assert encode_native_board(board, 4)[0, 109] == np.float32(1 / 100)


def test_native_ep_cutoff_and_pseudo_legal_not_legal_capture():
    board = position("e2e4")
    assert board.ep_square is not None and native_ep_square(board) is None
    board = position("e2e4", "g7g6", "e4e5", "f7f5")
    planes = encode_native_board(board)
    assert np.flatnonzero(planes[:, 108]).tolist() == [chess.F8]
    assert not planes[:, 13:104].any()
    board.push_uci("g1f3")
    assert not encode_native_board(board)[:, 13:104].any()
    assert not encode_native_board(board)[:, 108].any()
    board.push_uci("g8f6")
    planes = encode_native_board(board)
    assert planes[:, 13:25].any() and not planes[:, 26:104].any()
    pinned = position("f7f5", fen="k3r3/5p2/8/4P3/8/8/8/4K3 b - - 0 1")
    assert pinned.has_pseudo_legal_en_passant() and not pinned.has_legal_en_passant()
    assert native_ep_square(pinned) == chess.F6


def test_native_history_castling_and_repeat_only():
    board = position("e2e4", "e7e5", "e1e2", "e8e7", "e2d3")
    assert canonical_transform(board) == 0
    planes = encode_native_board(board)
    assert planes[12, 5] == planes[43, 11] == 1
    assert planes[:, 13:25].any() and not planes[:, 26:104].any()
    board.push_uci("e7e6")
    assert canonical_transform(board) == 1
    assert encode_native_board(board)[20, 5] == 1
    repeated = position("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6")
    x = encode_native_board(repeated, 5)
    assert np.all(x[:, 12] == 1) and np.all(x[:, 25] == 1)
    assert np.all(x[:, 38] == 1)
    assert not x[:, 39:104].any()


def test_native_root_fen_ep_preserved_and_one_prior_pawn_move_inferred():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    assert native_ep_square(board) == chess.E3
    old = encode_native_board(board, 1)
    assert old[chess.E7, 19] == 1 and old[chess.E5, 19] == 0
    assert not old[:, 26:104].any()
    canonical = encode_native_board(board, 4)
    assert canonical[chess.E8, 108] == 1 and not canonical[:, 13:104].any()


def test_native_action_ids_castling_promotions_and_canonical_files():
    for turn in ("w", "b"):
        board = chess.Board(f"r3k2r/8/8/8/8/8/8/R3K2R {turn} KQkq - 0 1")
        legal = native_legal_moves_by_id(board)
        assert board.is_kingside_castling(legal[ACTION_TO_ID[4, 7, 0]])
        assert board.is_queenside_castling(legal[ACTION_TO_ID[4, 0, 0]])
    for fen in ("7k/P7/8/8/8/8/8/K7 w - - 0 1", "k7/8/8/8/8/8/p7/7K b - - 0 1"):
        board = chess.Board(fen)
        legal = native_legal_moves_by_id(board)
        assert len(legal) == board.legal_moves.count()
        assert {m.promotion for m in legal.values() if m.promotion} == {2, 3, 4, 5}
    board = chess.Board("7k/8/8/8/8/8/P7/K7 w - - 0 1")
    assert canonical_transform(board) == 1
    assert native_legal_moves_by_id(board)[ACTION_TO_ID[15, 23, 0]].uci() == "a2a3"


def test_native_policy_inference_uses_native_inputs_and_action_map():
    class Model(torch.nn.Module):
        input_encoding = "lc0-training-v1:4"

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, inputs):
            np.testing.assert_array_equal(inputs[0].numpy(), encode_native_board(board))
            logits = torch.zeros((len(inputs), 1858))
            logits[:, ACTION_TO_ID[4, 7, 0]] = 10
            return {"policy": logits}

    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert choose_move(Model(), board).uci() == "e1g1"
    with pytest.raises(ValueError, match="incompatible"):
        encoding_functions("lc0-training-v1:6")
