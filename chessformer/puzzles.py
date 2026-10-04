"""Complete-line puzzle scoring with ChessBench PGN history and mate alternatives."""

import argparse
import csv
import io
import json
import math
from pathlib import Path

import chess
import chess.pgn
import torch

from .data import sha256_file
from .play import choose_move, load_model


def solve_board(board, moves, pick_move, scoring="chessbench"):
    """Lichess lists an opponent setup move, then alternating solver/reply moves.

    Validate the entire supplied line first. Every solver move must match unless
    ChessBench scoring accepts an alternative move that immediately checkmates.
    """
    if scoring not in ("chessbench", "exact"):
        raise ValueError("unknown puzzle scoring convention")
    moves = [chess.Move.from_uci(m) for m in moves.split()]
    if len(moves) < 2:
        raise ValueError("puzzle line needs setup and solution moves")
    board = board.copy(stack=True)
    if board.chess960 or not board.is_valid():
        raise ValueError("puzzles require a valid standard-chess position")
    check = board.copy(stack=True)
    for move in moves:
        if move not in check.legal_moves:
            raise ValueError(f"illegal puzzle solution move {move.uci()}")
        check.push(move)
    board.push(moves[0])
    for i, expected in enumerate(moves[1:], 1):
        if i % 2 == 1:
            predicted = pick_move(board.copy(stack=True))
            if not isinstance(predicted, chess.Move) or predicted not in board.legal_moves:
                raise ValueError(f"agent returned an illegal puzzle move: {predicted}")
            if predicted != expected:
                board.push(predicted)
                return scoring == "chessbench" and board.is_checkmate()
        board.push(expected)
    return True


def solve_line(fen, moves, pick_move, scoring="chessbench"):
    return solve_board(chess.Board(fen), moves, pick_move, scoring)


def puzzle_board(row, input_format):
    if input_format == "lichess":
        return chess.Board(row["FEN"])
    if input_format != "chessbench":
        raise ValueError("unknown puzzle input format")
    if not row.get("PGN", "").strip():
        raise ValueError("ChessBench puzzle requires PGN history")
    stream = io.StringIO(row["PGN"])
    game = chess.pgn.read_game(stream)
    if game is None or game.errors or chess.pgn.read_game(stream) is not None:
        raise ValueError("puzzle PGN must contain one correctly parsed game")
    board = game.end().board()
    if row.get("FEN") and chess.Board(row["FEN"]).fen() != board.fen():
        raise ValueError("puzzle PGN history disagrees with its FEN")
    return board


def accuracy_summary(solved, count):
    """Descriptive 95% Wilson interval; not a playing-strength estimate."""
    if not count:
        raise ValueError("accuracy needs at least one puzzle")
    p, z = solved / count, 1.959963984540054
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return {"puzzles": count, "solved": solved, "accuracy": p,
            "wilson_95": [max(0., center - radius), min(1., center + radius)]}


def evaluate_csv(path, pick_move, limit=None, input_format="auto", scoring="chessbench", rating_bin_width=100):
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if input_format not in ("auto", "chessbench", "lichess") or rating_bin_width < 1:
        raise ValueError("invalid puzzle format or rating bin width")
    details, seen = [], set()
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        if not {"Moves", "Rating"} <= columns:
            raise ValueError("puzzle CSV requires Moves and Rating columns")
        chosen_format = ("chessbench" if "PGN" in columns else "lichess") if input_format == "auto" else input_format
        required = "PGN" if chosen_format == "chessbench" else "FEN"
        if required not in columns:
            raise ValueError(f"puzzle CSV requires {required}")
        for index, row in enumerate(reader):
            # ChessBench's evaluator uses dataframe row indices; some exports
            # retain PuzzleId while others omit it. Record both when available.
            identity = row.get("PuzzleId", str(index))
            if not identity:
                raise ValueError("empty PuzzleId")
            if identity in seen:
                raise ValueError(f"duplicate PuzzleId: {identity}")
            seen.add(identity)
            board = puzzle_board(row, chosen_format)
            solved = solve_board(board, row["Moves"], pick_move, scoring)
            details.append({"id": identity, "row": index, "rating": int(row["Rating"]), "solved": solved,
                            "history_plies": len(board.move_stack)})
            if limit is not None and len(details) >= limit:
                break
    if not details:
        raise ValueError("empty puzzle set")
    solved = sum(row["solved"] for row in details)
    bins = {}
    for row in details:
        low = row["rating"] // rating_bin_width * rating_bin_width
        bins.setdefault(low, []).append(row["solved"])
    return {"dataset_sha256": sha256_file(path), "input_format": chosen_format, "scoring": scoring,
            "limit": limit, "rating_bin_width": rating_bin_width, **accuracy_summary(solved, len(details)),
            "rating_bins": [{"low": low, "high_exclusive": low + rating_bin_width,
                             **accuracy_summary(sum(values), len(values))} for low, values in sorted(bins.items())],
            "details": details}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--strategy", choices=["policy", "value"], default="policy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--input-format", choices=["auto", "chessbench", "lichess"], default="auto")
    parser.add_argument("--scoring", choices=["chessbench", "exact"], default="chessbench")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError("puzzle report exists; choose a new --out")
    torch.set_num_threads(args.threads)
    model = load_model(args.checkpoint, args.device)
    result = evaluate_csv(args.csv, lambda board: choose_move(model, board, args.strategy), args.limit,
                          args.input_format, args.scoring)
    result.update(checkpoint_sha256=sha256_file(args.checkpoint), strategy=args.strategy)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2))
