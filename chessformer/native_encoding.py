"""Lc0 board/action conventions for inference on native training checkpoints.

Matches the pinned engine's standard-chess encoder with FillEmptyHistory::NO,
then uses the trainer's rule-50 scaling. Native chunks already contain history;
their producer's empty-history option cannot be recovered from the record.
"""

import chess
import numpy as np

from .encoding import ACTION_TO_ID, orient_square
from .native import INPUT_FORMATS


def transform_square(square, transform):
    if transform & 1:
        square ^= 7
    if transform & 2:
        square ^= 56
    if transform & 4:
        square = chess.square(7 - chess.square_rank(square), 7 - chess.square_file(square))
    return square


def _check(board, input_format):
    if input_format not in INPUT_FORMATS:
        raise ValueError(f"unsupported native input format: {input_format}")
    if board.chess960:
        raise ValueError("native inference currently supports standard chess only")
    if board.king(board.turn) is None:
        raise ValueError("native encoding requires a side-to-move king")


def canonical_transform(board, input_format=4):
    _check(board, input_format)
    if input_format < 3 or board.clean_castling_rights():
        return 0
    king = orient_square(board.king(board.turn), board.turn)
    transform = int(chess.square_file(king) < 4)
    king = transform_square(king, transform)
    if board.pawns:
        return transform
    if chess.square_rank(king) >= 4:
        transform |= 2
        king ^= 56
    diagonal = chess.square_file(king) + chess.square_rank(king)
    if diagonal > 7:
        return transform | 4
    if diagonal == 7:
        for mask in (board.occupied, board.occupied_co[board.turn], board.kings,
                     board.queens, board.rooks, board.knights, board.bishops):
            squares = [transform_square(orient_square(s, board.turn), transform)
                       for s in chess.scan_forward(mask)]
            value = sum(1 << s for s in squares)
            alternative = sum(1 << transform_square(s, 4) for s in squares)
            if value != alternative:
                return transform | (4 if value > alternative else 0)
    return transform


def native_ep_square(board):
    # Lc0 preserves explicit FEN EP fields, but creates them after a move only
    # if an opposing pawn can pseudo-capture (even when that pawn is pinned).
    if not board.move_stack or board.has_pseudo_legal_en_passant():
        return board.ep_square
    return None


def _repetition_key(board):
    return (board.pawns, board.knights, board.bishops, board.rooks, board.queens,
            board.kings, board.occupied_co[chess.WHITE], board.turn,
            board.clean_castling_rights(), native_ep_square(board))


def _history(board):
    # Keep enough earlier positions for repetition flags in every emitted slot.
    # Stackless copies preserve the native EP decision separately below.
    current = board.copy(stack=True)
    snapshots = []
    while True:
        snapshots.append((current.copy(stack=False), native_ep_square(current),
                          _repetition_key(current)))
        if not current.move_stack:
            break
        current.pop()
    counts, result = {}, []
    for position, ep, key in reversed(snapshots):
        if position.halfmove_clock == 0:
            counts.clear()
        repetitions = counts.get(key, 0)
        counts[key] = repetitions + 1
        result.append((position, ep, repetitions > 0))
    return result[::-1]


def encode_native_board(board, input_format=4):
    """Return trainer-scaled [64,112] float32 planes for the supplied history."""
    transform = canonical_transform(board, input_format)
    x = np.zeros((64, 112), dtype=np.float32)
    history = _history(board)
    canonical = input_format >= 3
    repeat_only = input_format in (5, 133)
    castling = board.clean_castling_rights()
    slot = 0
    for age, (position, ep, repeated) in enumerate(history):
        if canonical and (position.clean_castling_rights() != castling or (age and ep is not None)):
            break
        if slot == 0 or not repeat_only or repeated:
            for square, piece in position.piece_map().items():
                channel = slot * 13 + (0 if piece.color == board.turn else 6) + piece.piece_type - 1
                x[orient_square(square, board.turn), channel] = 1
            x[:, slot * 13 + 12] = repeated
            slot += 1
        if slot == 8 or (canonical and position.halfmove_clock == 0):
            break
    else:
        # With NO fill, the engine still reconstructs one preceding pawn move
        # when the root FEN has an EP flag. Canonical history stops before this.
        position, ep, repeated = history[-1]
        if not canonical and ep is not None and slot < 8:
            prior = position.copy(stack=False)
            direction = 8 if position.turn == chess.BLACK else -8
            pawn = prior.remove_piece_at(ep + direction)
            if pawn is None or pawn.piece_type != chess.PAWN:
                raise ValueError("invalid FEN en-passant pawn")
            prior.set_piece_at(ep - direction, pawn)
            for square, piece in prior.piece_map().items():
                channel = slot * 13 + (0 if piece.color == board.turn else 6) + piece.piece_type - 1
                x[orient_square(square, board.turn), channel] = 1
            x[:, slot * 13 + 12] = repeated
    for own, color in enumerate((board.turn, not board.turn)):
        for kingside in (False, True):
            right = (board.has_kingside_castling_rights(color) if kingside
                     else board.has_queenside_castling_rights(color))
            if not right:
                continue
            if input_format == 1:
                x[:, 104 + 2 * own + int(kingside)] = 1
            else:
                square = chess.square(7 if kingside else 0, 0 if color else 7)
                x[orient_square(square, board.turn), 104 + int(kingside)] = 1
    if canonical:
        ep = native_ep_square(board)
        if ep is not None:
            x[56 + chess.square_file(ep), 108] = 1
    else:
        x[:, 108] = board.turn == chess.BLACK
    x[:, 109] = board.halfmove_clock / (100 if input_format >= 4 else 99)
    x[:, 110] = input_format in (132, 133) and board.turn == chess.BLACK
    x[:, 111] = 1
    if transform:
        permutation = [transform_square(s, transform) for s in range(64)]
        x[permutation, :109] = x[:, :109].copy()
    return x


def native_legal_moves_by_id(board, input_format=4):
    transform = canonical_transform(board, input_format)
    result = {}
    for move in board.legal_moves:
        destination = move.to_square
        if board.is_castling(move):
            destination = chess.square(7 if board.is_kingside_castling(move) else 0,
                                       chess.square_rank(move.from_square))
        src, dst = [transform_square(orient_square(s, board.turn), transform)
                    for s in (move.from_square, destination)]
        promotion = 0 if move.promotion in (None, chess.KNIGHT) else move.promotion
        result[ACTION_TO_ID[src, dst, promotion]] = move
    return result
