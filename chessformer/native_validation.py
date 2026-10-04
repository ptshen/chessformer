"""Validate native records as consecutive standard-chess positions.

Reconstruct the current board, undoing the stored canonical transform, and replay
the recorded played_idx. This checks interior chronology, not corpus provenance
or whether a chunk's beginning/end was removed before it reached the importer.
"""

import struct

import chess
import numpy as np

from .encoding import orient_square
from .native import record_header
from .native_encoding import canonical_transform, native_ep_square, native_legal_moves_by_id, transform_square


def _pieces(raw, slot, turn, transform):
    planes = np.unpackbits(np.frombuffer(raw, dtype=np.uint8, count=96, offset=7440 + slot * 104)).reshape(12, 64)
    inverse = {transform_square(square, transform): square for square in range(64)}
    board = chess.Board(None)
    board.turn = turn
    for channel, mask in enumerate(planes):
        color = turn if channel < 6 else not turn
        for stored in np.flatnonzero(mask):
            square = orient_square(inverse[int(stored)], turn)
            if board.piece_at(square) is not None:
                raise ValueError("native piece planes overlap")
            board.set_piece_at(square, chess.Piece(channel % 6 + 1, color))
    return board


def current_board(raw, preceding_position=None):
    """Decode current pieces/rights/clock; a preceding position supplies legacy EP.

`preceding_position` is the expected board AFTER replaying the prior frame's move.
Legacy formats omit EP; at a chunk's first frame it is inferred from slot 1 when
that slot records the opponent's preceding double pawn push.
"""
    _, fmt = record_header(raw)
    if fmt < 3 and raw[8276] not in (0, 1):
        raise ValueError("native side-to-move flag must be 0 or 1")
    transform = (raw[8278] & 7) if fmt >= 3 else 0
    turn = not bool(raw[8278] & 128) if fmt >= 3 else not bool(raw[8276])
    board = _pieces(raw, 0, turn, transform)
    board.halfmove_clock = raw[8277]
    for i, mask in enumerate(raw[8272:8276]):
        if fmt == 1:
            if mask not in (0, 1):
                raise ValueError("classical castling flag must be 0 or 1")
            mask = (1 << (7 if i % 2 else 0)) if mask else 0
        if mask not in (0, 1 << (7 if i % 2 else 0)):
            raise ValueError("native chronology validation supports standard-chess castling only")
        color = turn if i < 2 else not turn
        board.castling_rights |= mask << (0 if color else 56)
    if fmt >= 3:
        mask = raw[8276]
        if mask and (mask & (mask - 1) or transform & 6):
            raise ValueError("invalid canonical en-passant flags")
        if mask:
            file = mask.bit_length() - 1
            if transform & 1:
                file = 7 - file
            board.ep_square = chess.square(file, 5 if turn else 2)
    elif preceding_position is not None:
        board.ep_square = preceding_position.ep_square
    else:
        previous = _pieces(raw, 1, turn, 0)
        removed, added = previous.occupied & ~board.occupied, board.occupied & ~previous.occupied
        if chess.popcount(removed) == chess.popcount(added) == 1:
            src, dst = chess.msb(removed), chess.msb(added)
            pawn = chess.Piece(chess.PAWN, not turn)
            if previous.piece_at(src) == board.piece_at(dst) == pawn and dst - src == (-16 if turn else 16):
                candidate = previous.copy(stack=False)
                candidate.remove_piece_at(src)
                candidate.set_piece_at(dst, pawn)
                if candidate.piece_map() == board.piece_map():
                    board.ep_square = (src + dst) // 2
    if not board.is_valid():
        raise ValueError(f"invalid native standard-chess position (status={int(board.status())})")
    if canonical_transform(board, fmt) != transform:
        raise ValueError("native canonical transform disagrees with current position")
    return board


def position_key(board, include_ep=True):
    return (board.board_fen(), board.turn, board.castling_rights, board.halfmove_clock,
            native_ep_square(board) if include_ep else None)


class ChronologyValidator:
    def __init__(self):
        self.expected = None
        self.transitions = 0

    def accept(self, raw):
        _, fmt = record_header(raw)
        board = current_board(raw, self.expected)
        if self.expected is not None:
            if position_key(board, fmt >= 3) != position_key(self.expected, fmt >= 3):
                raise ValueError("nonconsecutive native positions; recorded played_idx does not reach the next frame")
            self.transitions += 1
        index = struct.unpack_from("<H", raw, 8344)[0]
        legal = native_legal_moves_by_id(board, fmt)
        if index not in legal:
            raise ValueError(f"native played_idx {index} is not legal in its recorded position")
        board.push(legal[index])
        self.expected = board
