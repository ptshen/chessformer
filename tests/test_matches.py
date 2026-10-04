import json

import chess
import chess.engine
import chess.pgn
import pytest

from chessformer.matches import MatchConfig, adjudicated_result, play_game, run_match, score_summary


class FirstLegal:
    def __init__(self):
        self.seen = []

    def play(self, board):
        self.seen.append(board.copy(stack=True))
        return next(iter(board.legal_moves))


def test_adjudication_sign_mate_and_strict_threshold():
    assert adjudicated_result(chess.engine.PovScore(chess.engine.Cp(1300), chess.WHITE)) is None
    assert adjudicated_result(chess.engine.PovScore(chess.engine.Cp(-1300), chess.BLACK)) is None
    assert adjudicated_result(chess.engine.PovScore(chess.engine.Cp(1301), chess.BLACK)) == "0-1"
    assert adjudicated_result(chess.engine.PovScore(chess.engine.Cp(-1301), chess.BLACK)) == "1-0"
    assert adjudicated_result(chess.engine.PovScore(chess.engine.Mate(3), chess.BLACK)) == "0-1"
    assert adjudicated_result(chess.engine.PovScore(chess.engine.Mate(-3), chess.BLACK)) == "1-0"


def test_opening_history_black_turn_and_color_pairing(tmp_path):
    book = tmp_path / "openings.pgn"
    board = chess.Board()
    board.push_uci("e2e4")
    book.write_text(str(chess.pgn.Game.from_board(board)) + "\n\n")
    agents = FirstLegal(), FirstLegal()
    score = lambda board: chess.engine.PovScore(chess.engine.Cp(1400), chess.WHITE)
    result = run_match(agents, ("a", "b"), book, tmp_path / "match", MatchConfig(games=2), score)
    assert len(agents[0].seen) == len(agents[1].seen) == 1
    assert all(a.seen[0].move_stack == board.move_stack and a.seen[0].turn == chess.BLACK for a in agents)
    assert result["summary"]["wins"] == result["summary"]["losses"] == 1
    assert result["summary"]["score_rate"] == 0.5
    with (tmp_path / "match/games.pgn").open() as stream:
        games = [chess.pgn.read_game(stream), chess.pgn.read_game(stream)]
    assert [g.headers["White"] for g in games] == ["a", "b"]
    assert all(list(g.mainline_moves())[0].uci() == "e2e4" for g in games)
    assert json.loads((tmp_path / "match/manifest.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        run_match(agents, ("a", "b"), book, tmp_path / "match", MatchConfig(games=2), score)


def test_natural_mate_draw_claim_and_ply_limit_are_distinct():
    board = chess.Board()
    for move in ("f2f3", "e7e5", "g2g4"):
        board.push_uci(move)
    class Mate:
        def play(self, board):
            return chess.Move.from_uci("d8h4")
    _, result = play_game((Mate(), Mate()), ("a", "b"), 0, board, MatchConfig(games=2, protocol="rules"))
    assert result["result"] == "0-1" and result["termination"] == "checkmate"
    drawn = chess.Board("8/8/8/8/8/2k5/6r1/K7 w - - 99 1")
    agents = FirstLegal(), FirstLegal()
    _, result = play_game(agents, ("a", "b"), 0, drawn, MatchConfig(games=2, protocol="rules"))
    assert result["result"] == "1/2-1/2" and not any(a.seen for a in agents)
    _, result = play_game(agents, ("a", "b"), 0, chess.Board(), MatchConfig(games=2, protocol="rules", max_plies=1))
    assert result["result"] == "*" and result["termination"] == "ply_limit"


def test_summary_requires_complete_color_pairs_and_keeps_sampling_units():
    records = [{"pair": i, "white_index": w, "result": "1-0" if w == 0 else "0-1"}
               for i in range(500) for w in (0, 1)]
    result = score_summary(records)
    assert result["score_rate"] == 1 and result["wins"] == 1000
    assert result["score_hoeffding_95"][0] < 1
    assert result["logistic_elo_difference"]["upper_unbounded"]
    assert score_summary(records[:-1])["score_rate"] is None
    records[-1]["result"] = "*"
    assert score_summary(records)["score_rate"] is None
    assert score_summary([{"pair": 0, "white_index": 0, "result": "1-0"}] * 2)["score_rate"] is None


def test_illegal_agent_marks_match_failed_and_requires_adjudicator(tmp_path):
    class Illegal:
        def play(self, board):
            return chess.Move.from_uci("e2e5")
    book = tmp_path / "book.pgn"
    book.write_text(str(chess.pgn.Game()) + "\n\n")
    with pytest.raises(ValueError, match="requires a Stockfish"):
        run_match((Illegal(), FirstLegal()), ("a", "b"), book, tmp_path / "needs_engine", MatchConfig(games=2))
    with pytest.raises(ValueError, match="illegal move"):
        run_match((Illegal(), FirstLegal()), ("a", "b"), book, tmp_path / "failed", MatchConfig(games=2, protocol="rules"))
    assert json.loads((tmp_path / "failed/manifest.json").read_text())["status"] == "failed"


def test_evaluation_against_verified_chessbench_source():
    from chessformer.evaluation_reference import verify_evaluation
    try:
        report = verify_evaluation()
    except FileNotFoundError:
        pytest.skip("optional pinned sources unavailable; run python3 -m chessformer.reference --fetch")
    assert report["puzzle_cases"] == 29 and report["match_cases"] == 24
