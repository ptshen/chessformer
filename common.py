"""Shared encoding logic: FEN tokenizer, UCI action space, Stockfish win-prob conversion.

Follows "Grandmaster-Level Chess Without Search" (arXiv 2402.04494):
- FEN flattened to a fixed 77-token representation.
- Actions indexed into a lexicographically sorted list of 1968 possible UCI moves.
- Centipawn scores converted to win probability, discretized into K bins,
  trained with HL-Gauss smoothed targets.
"""

import math

import numpy as np

SEQ_LEN = 77  # board 64 + side 1 + castling 4 + en passant 2 + halfmove 3 + fullmove 3

_CHARS = sorted(set("pnbrqkPNBRQK") | set("0123456789") | set("abcdefgh") | set("w-."))
CHAR_TO_ID = {c: i for i, c in enumerate(_CHARS)}
VOCAB_SIZE = len(_CHARS)

NUM_BINS = 128
BIN_CENTERS = (np.arange(NUM_BINS) + 0.5) / NUM_BINS
HL_GAUSS_SIGMA = 0.75 / NUM_BINS


def tokenize_fen(fen: str) -> np.ndarray:
    """Encode a FEN string as a fixed-length array of 77 uint8 token ids."""
    board, side, castling, ep, halfmove, fullmove = fen.split(" ")
    flat_board = ""
    for c in board:
        if c == "/":
            continue
        flat_board += "." * int(c) if c.isdigit() else c
    parts = (
        flat_board
        + side
        + castling.replace("-", "").ljust(4, ".")
        + ep.replace("-", "").ljust(2, ".")
        + halfmove.ljust(3, ".")
        + fullmove[:3].ljust(3, ".")
    )
    assert len(parts) == SEQ_LEN, fen
    return np.array([CHAR_TO_ID[c] for c in parts], dtype=np.uint8)


def _all_uci_moves() -> list[str]:
    """All 1968 UCI moves possible on some board: queen/knight geometry + promotions."""
    files = "abcdefgh"
    moves = set()
    for f in range(8):
        for r in range(8):
            frm = files[f] + str(r + 1)
            rays = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for df, dr in rays:
                for dist in range(1, 8):
                    tf, tr = f + df * dist, r + dr * dist
                    if 0 <= tf < 8 and 0 <= tr < 8:
                        moves.add(frm + files[tf] + str(tr + 1))
            for df, dr in [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]:
                tf, tr = f + df, r + dr
                if 0 <= tf < 8 and 0 <= tr < 8:
                    moves.add(frm + files[tf] + str(tr + 1))
    for f in range(8):  # promotions, incl. diagonal captures
        for df in (-1, 0, 1):
            tf = f + df
            if not 0 <= tf < 8:
                continue
            for promo in "nbrq":
                moves.add(files[f] + "7" + files[tf] + "8" + promo)
                moves.add(files[f] + "2" + files[tf] + "1" + promo)
    return sorted(moves)


ALL_MOVES = _all_uci_moves()
NUM_ACTIONS = len(ALL_MOVES)
MOVE_TO_ID = {m: i for i, m in enumerate(ALL_MOVES)}


def cp_to_win_prob(cp: float) -> float:
    """Stockfish centipawns (side-to-move perspective) -> win probability in [0, 1]."""
    return 1.0 / (1.0 + math.exp(-0.00368208 * max(-4000.0, min(4000.0, cp))))


def hl_gauss_targets(win_probs: np.ndarray) -> np.ndarray:
    """HL-Gauss soft labels [N, NUM_BINS]: Gaussian around each label, integrated per bin."""
    from scipy.stats import norm

    edges = np.linspace(0.0, 1.0, NUM_BINS + 1)
    cdf = norm.cdf(edges[None, :], loc=win_probs[:, None], scale=HL_GAUSS_SIGMA)
    probs = np.diff(cdf, axis=1)
    return probs / probs.sum(axis=1, keepdims=True)
