"""Compare evaluation behavior with verified, unmodified ChessBench functions.

Uses synthetic legal positions and scripted agents/scores. Does not run a real
Stockfish binary, BayesElo, reference neural weights, or the curated datasets.
"""

import argparse
import ast
from contextlib import redirect_stdout
import datetime
import io
import json
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np

from .data import sha256_file
from .matches import MatchConfig, REFERENCE_COMMIT, play_game
from .puzzles import puzzle_board, solve_board
from .reference import check_sources


def source_functions(specification, path, names):
    entry = next(e for e in specification["files"] if e["repository"] == "google-deepmind/searchless_chess"
                 and e["commit"] == REFERENCE_COMMIT and e["path"] == path)
    tree = ast.parse(Path(entry["local_path"]).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError("reference functions missing")
    # Defer annotations so pandas/JAX/absl and neural checkpoints are unnecessary.
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    namespace = {"chess": chess, "io": io, "datetime": datetime}
    exec(compile(module, entry["local_path"], "exec"), namespace)
    return namespace


class ScriptedAgent:
    def __init__(self, moves=()):
        self.moves = iter(moves)
        self.seen = []

    def play(self, board):
        self.seen.append({"fen": board.fen(), "history": [m.uci() for m in board.move_stack]})
        move = next(self.moves, None)
        return chess.Move.from_uci(move) if move is not None else next(iter(board.legal_moves))


class ScriptedAdjudicator:
    def __init__(self, scores):
        self.scores = iter(scores)

    def analyse(self, board):
        return {"score": chess.engine.PovScore(next(self.scores, chess.engine.Cp(0)), board.turn)}


def position(moves=(), fen=chess.STARTING_FEN):
    board = chess.Board(fen)
    for move in moves:
        board.push_uci(move)
    return board


def verify_evaluation(seed=83):
    specification = check_sources()
    puzzle_source = source_functions(specification, "src/puzzles.py",
                                    ["evaluate_puzzle_from_pandas_row", "evaluate_puzzle_from_board"])
    tournament_source = source_functions(specification, "src/tournament.py", ["_play_game"])
    cases = [
        ("complete_line", position(["e2e4"]), "e7e5 g1f3 b8c6 f1b5", ["g1f3", "f1b5"]),
        ("late_failure", position(["e2e4"]), "e7e5 g1f3 b8c6 f1b5", ["g1f3", "f1c4"]),
        ("early_failure", position(), "e2e4 e7e5 g1f3 b8c6", ["e7e6"]),
        ("alternative_mate", position(fen="7k/p4K2/4Q3/8/8/8/8/8 b - - 0 1"),
         "a7a6 e6h6", ["e6h3"]),
    ]
    mate = cases[-1]
    def mirror(move):
        move = chess.Move.from_uci(move)
        return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square),
                          promotion=move.promotion).uci()
    cases.append(("alternative_mate_black", mate[1].mirror(),
                  " ".join(mirror(m) for m in mate[2].split()), [mirror(m) for m in mate[3]]))
    rng = np.random.default_rng(seed)
    for i in range(24):
        board = chess.Board()
        for _ in range(int(rng.integers(0, 30))):
            if board.is_game_over():
                break
            legal = list(board.legal_moves)
            board.push(legal[int(rng.integers(len(legal)))])
        check, moves, predictions = board.copy(stack=True), [], []
        for j in range(6):
            if check.is_game_over():
                break
            legal = list(check.legal_moves)
            move = legal[int(rng.integers(len(legal)))]
            moves.append(move.uci())
            if j % 2:
                alternative = next((m for m in legal if m != move), move)
                predictions.append((alternative if i % 2 and j == 3 else move).uci())
            check.push(move)
        if len(moves) >= 2:
            cases.append((f"generated_{i}", board, " ".join(moves), predictions))
    puzzle_results = []
    for name, board, moves, predictions in cases:
        row = {"PGN": str(chess.pgn.Game.from_board(board)), "FEN": board.fen(), "Moves": moves}
        reference, actual = ScriptedAgent(predictions), ScriptedAgent(predictions)
        expected = puzzle_source["evaluate_puzzle_from_pandas_row"](row, reference)
        observed = solve_board(puzzle_board(row, "chessbench"), moves, actual.play)
        assert observed == expected, name
        assert actual.seen == reference.seen, name
        puzzle_results.append({"case": name, "solved": observed, "decisions": len(actual.seen),
                               "initial_history_plies": len(board.move_stack)})

    matches = [
        ("natural_checkmate", position(["f2f3", "e7e5", "g2g4"]), ["d8h4"], []),
        ("fifty_move_claim", position(fen="8/8/8/8/8/2k5/6r1/K7 w - - 99 1"), [], []),
        ("current_threefold", position(["g1f3", "g8f6", "f3g1", "f6g8"] * 2), [], []),
        ("prospective_threefold", position(["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]),
         ["f6g8"], []),
    ]
    for initial_black in (False, True):
        board = position(["e2e4"] if initial_black else [])
        for sign in (-1, 1):
            matches.append((f"strict_cp_threshold_{initial_black}_{sign}", board, [],
                            [chess.engine.Cp(sign * 1300), chess.engine.Cp(sign * 1301)]))
            matches.append((f"mate_score_{initial_black}_{sign}", board, [], [chess.engine.Mate(sign * 3)]))
    match_results = []
    for name, board, moves, scores in matches:
        for white_index in (0, 1):
            agents = tuple(ScriptedAgent(moves) for _ in range(2))
            reference_agents = tuple(ScriptedAgent(moves) for _ in range(2))
            adjudicator = ScriptedAdjudicator(scores)
            tournament_source.update(_EVAL_STOCKFISH_ENGINE=ScriptedAdjudicator(scores), _MIN_SCORE_TO_STOP=1300)
            with redirect_stdout(io.StringIO()):
                expected = tournament_source["_play_game"](reference_agents, ("a", "b"),
                                                          ("a", "b")[white_index], board.copy(stack=True))
            observed, record = play_game(agents, ("a", "b"), white_index, board,
                                         MatchConfig(games=2, max_plies=8),
                                         lambda b: adjudicator.analyse(b)["score"])
            for key in ("White", "Black", "Result"):
                assert observed.headers[key] == expected.headers[key], (name, key)
            assert observed.end().board().fen() == expected.end().board().fen(), name
            assert list(observed.mainline_moves()) == list(expected.mainline_moves()), name
            assert [a.seen for a in agents] == [a.seen for a in reference_agents], name
            match_results.append({"case": name, "white_index": white_index, **record})
    return {"scope": "synthetic protocol conformance; no curated benchmark, neural reference, Stockfish, or BayesElo run",
            "reference_commit": REFERENCE_COMMIT, "source_files_verified": len(specification["files"]),
            "seed": seed, "puzzle_cases": len(puzzle_results), "match_cases": len(match_results),
            "puzzle_results": puzzle_results, "match_results": match_results,
            "intentional_differences": ["Reject malformed inputs and illegal agent moves",
                                        "Stop at a configured ply limit as incomplete, never a scored draw",
                                        "Use exact terminal outcomes without invoking the adjudicator"],
            "source_files_sha256": {str(p): sha256_file(p) for p in map(Path, [
                "chessformer/puzzles.py", "chessformer/matches.py", "chessformer/evaluation_reference.py"])}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/evaluation-conformance.json")
    args = parser.parse_args()
    result = verify_evaluation()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("puzzle_results", "match_results")}, indent=2))
