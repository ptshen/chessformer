"""Explicit local board/action convention, version cf112-v1.

This is not the binary Lc0 input/policy format. Black positions are rank-mirrored;
all historical pieces retain the CURRENT side-to-move's perspective.
"""

import chess
import numpy as np

ENCODING = "cf112-v1"


def orient_square(square: int, turn: chess.Color) -> int:
    return square if turn == chess.WHITE else square ^ 56


def _actions():
    traversals = []
    for src in range(64):
        for dst in range(64):
            df, dr = abs(src % 8 - dst % 8), abs(src // 8 - dst // 8)
            if src != dst and (df == 0 or dr == 0 or df == dr or (df, dr) in ((1, 2), (2, 1))):
                traversals.append((src, dst, 0))
    promotions = [(48 + f, 56 + t, p) for f in range(8) for t in range(8)
                  if abs(f - t) <= 1 for p in (chess.QUEEN, chess.ROOK, chess.BISHOP)]
    return tuple(traversals + promotions)


ACTIONS = _actions()
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)
assert NUM_ACTIONS == 1858


def move_id(move: chess.Move, turn: chess.Color) -> int:
    # Knight promotions share their traversal with a non-promoting move.
    promo = 0 if move.promotion in (None, chess.KNIGHT) else move.promotion
    return ACTION_TO_ID[(orient_square(move.from_square, turn),
                         orient_square(move.to_square, turn), promo)]


def legal_moves_by_id(board: chess.Board) -> dict[int, chess.Move]:
    if board.chess960:
        raise ValueError("cf112-v1 currently supports standard chess only")
    return {move_id(m, board.turn): m for m in board.legal_moves}


def encode_board(board: chess.Board) -> np.ndarray:
    """[64,112] float32; missing history is zero, never invented from a FEN.

    0:104 = eight groups of own PNBRQK, opponent PNBRQK, repetition.
    104/105 = queenside/kingside rook rights at their squares; 106/107 = 0;
    108 = EP file marker on rank 8; 109 = halfmove clock / 100;
    110 = 0; 111 = 1. History ends only when the supplied move stack does.
    """
    if board.chess960:
        raise ValueError("cf112-v1 currently supports standard chess only")
    x = np.zeros((64, 112), dtype=np.float32)
    current = board.copy(stack=True)
    for age in range(8):
        for square, piece in current.piece_map().items():
            channel = age * 13 + (0 if piece.color == board.turn else 6) + piece.piece_type - 1
            x[orient_square(square, board.turn), channel] = 1
        x[:, age * 13 + 12] = current.is_repetition(2)
        if not current.move_stack:
            break
        current.pop()
    for color in chess.COLORS:
        rank = 0 if color == chess.WHITE else 7
        if board.has_queenside_castling_rights(color):
            x[orient_square(chess.square(0, rank), board.turn), 104] = 1
        if board.has_kingside_castling_rights(color):
            x[orient_square(chess.square(7, rank), board.turn), 105] = 1
    if board.ep_square is not None:
        x[56 + chess.square_file(board.ep_square), 108] = 1
    x[:, 109] = board.halfmove_clock / 100
    x[:, 111] = 1
    return x


def board_from_record(record: dict) -> chess.Board:
    board = chess.Board(record.get("initial_fen", chess.STARTING_FEN))
    for uci in record.get("moves", []):
        board.push_uci(uci)
    return board
