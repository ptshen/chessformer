import chess
import chess.pgn
import csv
import pytest

from chessformer.puzzles import evaluate_csv, puzzle_board, solve_line


def test_every_solver_move_is_required_and_setup_is_not_scored():
    line = "e2e4 e7e5 g1f3 b8c6"
    seen = []
    def correct(board):
        seen.append(board.turn)
        return chess.Move.from_uci("e7e5" if len(board.move_stack) == 1 else "b8c6")
    assert solve_line(chess.STARTING_FEN, line, correct)
    assert seen == [chess.BLACK, chess.BLACK]
    assert not solve_line(chess.STARTING_FEN, line, lambda b: chess.Move.from_uci("e7e5" if len(b.move_stack) == 1 else "b8a6"))


def test_malformed_puzzle_is_not_silently_counted_as_failure():
    with pytest.raises(ValueError, match="illegal"):
        solve_line(chess.STARTING_FEN, "e2e4 e7e3", lambda b: None)


def test_chessbench_accepts_alternative_mate_but_exact_scoring_does_not():
    fen = "7k/p4K2/4Q3/8/8/8/8/8 b - - 0 1"
    pick = lambda board: chess.Move.from_uci("e6h3")
    assert solve_line(fen, "a7a6 e6h6", pick)
    assert not solve_line(fen, "a7a6 e6h6", pick, scoring="exact")
    with pytest.raises(ValueError, match="illegal puzzle move"):
        solve_line(fen, "a7a6 e6h6", lambda board: chess.Move.from_uci("e6f4"))


def test_csv_chessbench_preserves_history_and_records_protocol(tmp_path):
    board = chess.Board()
    board.push_uci("e2e4")
    row = {"PuzzleId": "history", "FEN": board.fen(), "Moves": "e7e5 g1f3 b8c6 f1b5",
           "Rating": 1520, "PGN": str(chess.pgn.Game.from_board(board))}
    path = tmp_path / "puzzles.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    histories = []
    def pick(board):
        histories.append([move.uci() for move in board.move_stack])
        return chess.Move.from_uci("g1f3" if len(histories) == 1 else "f1b5")
    result = evaluate_csv(path, pick)
    assert histories == [["e2e4", "e7e5"], ["e2e4", "e7e5", "g1f3", "b8c6"]]
    assert result["input_format"] == result["scoring"] == "chessbench"
    assert result["solved"] == 1 and result["details"][0]["history_plies"] == 1
    assert result["rating_bins"][0]["low"] == 1500
    assert result["wilson_95"][0] < 1 and result["wilson_95"][1] == 1
    with pytest.raises(ValueError, match="disagrees"):
        puzzle_board({**row, "FEN": chess.STARTING_FEN}, "chessbench")
    with pytest.raises(ValueError, match="requires PGN"):
        puzzle_board({**row, "PGN": ""}, "chessbench")


def test_csv_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "puzzles.csv"
    path.write_text('PuzzleId,FEN,Moves,Rating\na,"' + chess.STARTING_FEN + '",e2e4 e7e5,1200\n' +
                    'a,"' + chess.STARTING_FEN + '",d2d4 d7d5,1200\n')
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_csv(path, lambda board: next(iter(board.legal_moves)))
