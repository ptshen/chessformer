"""Paired opening-book matches following the pinned ChessBench stopping rules.

Produces PGN for BayesElo. A descriptive score/logistic-Elo summary is also saved;
it is not BayesElo and does not assign an absolute human or engine rating.
"""

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import datetime
import importlib
import json
import math
from pathlib import Path
import subprocess
import time

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

from .data import sha256_file
from .play import choose_move, load_model

REFERENCE_COMMIT = "4a41136164cbf79786eea628513579313e1fe454"


@dataclass(frozen=True)
class MatchConfig:
    games: int = 1000
    seed: int = 1
    protocol: str = "chessbench"
    adjudication_time: float = 0.01
    adjudication_cp: int = 1300
    max_plies: int = 2048

    def __post_init__(self):
        if self.games < 2 or self.games % 2:
            raise ValueError("games must be a positive even number")
        if self.protocol not in ("chessbench", "rules"):
            raise ValueError("unknown match protocol")
        if self.adjudication_time <= 0 or self.adjudication_cp <= 0 or self.max_plies < 1:
            raise ValueError("invalid adjudication or ply limit")


class ModelAgent:
    def __init__(self, checkpoint, strategy="policy", device="cpu"):
        self.model = load_model(checkpoint, device)
        self.strategy = strategy
        self.positions_evaluated = self.forward_calls = 0
        self.model.register_forward_pre_hook(self._count)
        self.provenance = {"kind": "chessformer", "checkpoint": str(checkpoint),
                           "checkpoint_sha256": sha256_file(checkpoint), "strategy": strategy,
                           "encoding": self.model.input_encoding, "device": str(device)}

    def _count(self, model, inputs):
        self.forward_calls += 1
        self.positions_evaluated += len(inputs[0])

    def play(self, board):
        return choose_move(self.model, board, self.strategy)


class UciAgent:
    def __init__(self, executable, move_time):
        self.engine = chess.engine.SimpleEngine.popen_uci(str(executable))
        self.limit = chess.engine.Limit(time=move_time)
        self.provenance = {"kind": "uci", "executable": str(executable),
                           "executable_sha256": sha256_file(executable), "id": self.engine.id,
                           "move_time_seconds": move_time}

    def play(self, board):
        return self.engine.play(board, self.limit).move

    def close(self):
        self.engine.quit()


def factory_agent(specification):
    """User-specified module:callable or module:mapping:key, e.g. ChessBench."""
    parts = specification.split(":")
    if len(parts) not in (2, 3):
        raise ValueError("factory must be module:callable or module:mapping:key")
    module = importlib.import_module(parts[0])
    factory = getattr(module, parts[1])
    if len(parts) == 3:
        factory = factory[parts[2]]
    agent = factory()
    if not callable(getattr(agent, "play", None)):
        raise ValueError("factory must return an agent with play(board)")
    return agent


def opening_boards(path):
    boards = []
    with open(path) as stream:
        while (game := chess.pgn.read_game(stream)) is not None:
            if game.errors:
                raise ValueError("opening PGN contains a parsing error")
            board = game.end().board()
            if board.chess960 or not board.is_valid() or board.is_game_over():
                raise ValueError("opening PGN contains an invalid or terminal standard-chess position")
            boards.append(board)
    if not boards:
        raise ValueError("opening book is empty")
    return boards


def adjudicated_result(score, threshold=1300):
    """PovScore from the side to move; threshold is strict, matching the source."""
    white = score.white()
    if not white.is_mate() and abs(white.score()) <= threshold:
        return None
    return "1-0" if white > chess.engine.Cp(0) else "0-1"


def play_game(agents, names, white_index, initial_board, cfg, adjudicate=None):
    if cfg.protocol == "chessbench" and adjudicate is None:
        raise ValueError("ChessBench protocol requires a Stockfish adjudicator")
    board = initial_board.copy(stack=True)
    played = 0
    adjudication_score = None
    while True:
        if board.is_game_over():
            result, termination = board.result(), board.outcome().termination.name.lower()
            break
        # The source claims a fifty-move draw if available now OR by announcing
        # a move, but claims repetition only when this board occurred three times.
        if board.can_claim_fifty_moves() or board.is_repetition(3):
            result, termination = "1/2-1/2", "claimable_draw"
            break
        if played >= cfg.max_plies:
            result, termination = "*", "ply_limit"
            break
        index = white_index if board.turn else 1 - white_index
        move = agents[index].play(board.copy(stack=True))
        if not isinstance(move, chess.Move) or move not in board.legal_moves:
            raise ValueError(f"{names[index]} returned illegal move {move}")
        board.push(move)
        played += 1
        # Exact terminal outcomes take precedence; the reference engine would
        # return the same result on these positions without a search decision.
        if cfg.protocol == "chessbench" and not board.is_game_over():
            score = adjudicate(board.copy(stack=True))
            result = adjudicated_result(score, cfg.adjudication_cp)
            if result is not None:
                termination, adjudication_score = "adjudication", str(score.white())
                break
    game = chess.pgn.Game.from_board(board)
    game.headers.update(Event="Chessformer replication", Date=datetime.date.today().strftime("%Y.%m.%d"),
                        White=names[white_index], Black=names[1 - white_index], Result=result,
                        Termination=termination)
    record = {"white_index": white_index, "result": result, "termination": termination,
              "played_plies": played, "opening_plies": len(initial_board.move_stack),
              "final_fen": board.fen(), "adjudication_score_white": adjudication_score}
    return game, record


def score_summary(records):
    incomplete = sum(r["result"] == "*" for r in records)
    scores, pairs = [], {}
    for row in records:
        if row["result"] == "*":
            continue
        white_score = {"1-0": 1., "0-1": 0., "1/2-1/2": 0.5}[row["result"]]
        score = white_score if row["white_index"] == 0 else 1 - white_score
        scores.append(score)
        pairs.setdefault(row["pair"], []).append((row["white_index"], score))
    counts = {"wins": scores.count(1.), "draws": scores.count(0.5), "losses": scores.count(0.),
              "incomplete_games": incomplete}
    if incomplete or not records or any(len(values) != 2 or {color for color, _ in values} != {0, 1}
                                        for values in pairs.values()):
        return {**counts, "score_rate": None, "reason": "requires every game and both colors for each opening"}
    pair_scores = [sum(score for _, score in values) / 2 for values in pairs.values()]
    mean = sum(pair_scores) / len(pair_scores)
    # Hoeffding's bound for [0,1] scores, with openings as sampling units.
    # Conservative even for all-win/all-draw samples where bootstrap degenerates.
    radius = math.sqrt(math.log(40) / (2 * len(pair_scores)))
    low, high = max(0., mean - radius), min(1., mean + radius)

    def elo(value):
        return 400 * math.log10(value / (1 - value)) if 0 < value < 1 else None

    return {**counts, "pairs": len(pairs), "score_rate": mean, "score_hoeffding_95": [low, high],
            "interval_unit": "randomly sampled opening pairs, not independent individual games",
            "logistic_elo_difference": {"estimate": elo(mean), "lower": elo(low), "upper": elo(high),
                                        "lower_unbounded": low == 0, "upper_unbounded": high == 1},
            "elo_method": "descriptive score-to-logistic conversion; not BayesElo or an absolute rating"}


def run_match(agents, names, openings_path, out, cfg=MatchConfig(), adjudicate=None, metadata=None):
    if len(names) != 2 or len(set(names)) != 2 or len(agents) != 2:
        raise ValueError("two agents with distinct names are required")
    if cfg.protocol == "chessbench" and adjudicate is None:
        raise ValueError("ChessBench protocol requires a Stockfish adjudicator")
    boards = opening_boards(openings_path)
    if len(boards) < cfg.games // 2:
        raise ValueError("opening book has fewer entries than the requested number of pairs")
    indices = np.random.default_rng(cfg.seed).choice(np.arange(len(boards)), size=cfg.games // 2, replace=False)
    out = Path(out)
    if out.exists():
        raise FileExistsError("match output exists; choose a new output directory")
    out.mkdir(parents=True)
    manifest = {"reference_commit": REFERENCE_COMMIT, "configuration": asdict(cfg), "agents": list(names),
                "opening_book_sha256": sha256_file(openings_path), "opening_indices": indices.tolist(),
                "metadata": metadata or {}, "status": "running"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records, started = [], time.perf_counter()
    initial_counts = [{key: getattr(a, key, 0) for key in ("positions_evaluated", "forward_calls")} for a in agents]
    try:
        with (out / "games.pgn").open("w") as pgn, (out / "games.jsonl").open("w") as log:
            for pair, opening_index in enumerate(indices):
                for white_index in (0, 1):
                    game, record = play_game(agents, names, white_index, boards[opening_index], cfg, adjudicate)
                    game.headers["Round"] = str(len(records) + 1)
                    game.headers["OpeningIndex"] = str(opening_index)
                    game.headers["Pair"] = str(pair)
                    record.update(pair=pair, opening_index=int(opening_index))
                    records.append(record)
                    pgn.write(str(game) + "\n\n")
                    pgn.flush()
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps({"game": len(records), **record}), flush=True)
    except BaseException:
        manifest["status"] = "failed"
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    counts = [{key: getattr(a, key) - initial_counts[i][key] if hasattr(a, key) else None
               for key in ("positions_evaluated", "forward_calls")} for i, a in enumerate(agents)]
    result = {"agents": list(names), "requested_games": cfg.games, "summary": score_summary(records),
              "elapsed_seconds": time.perf_counter() - started, "neural_evaluation_counts": counts,
              "pgn_sha256": sha256_file(out / "games.pgn"), "games_sha256": sha256_file(out / "games.jsonl")}
    manifest["status"] = "complete" if result["summary"]["incomplete_games"] == 0 else "incomplete"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_bayeselo(executable, out):
    out = Path(out).resolve()
    commands = "readpgn games.pgn\nelo\nmm\nexactdist\nratings\nx\nx\n"
    result = subprocess.run([str(Path(executable).resolve())], input=commands, text=True, capture_output=True,
                            cwd=out, check=True)
    (out / "bayeselo.txt").write_text(result.stdout + result.stderr)
    return {"binary_sha256": sha256_file(executable), "commands": commands, "output": "bayeselo.txt"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--strategy", choices=["policy", "value"], default="policy")
    opponent = parser.add_mutually_exclusive_group(required=True)
    opponent.add_argument("--opponent-checkpoint")
    opponent.add_argument("--opponent-uci")
    opponent.add_argument("--opponent-factory")
    parser.add_argument("--opponent-name", default="reference")
    parser.add_argument("--opponent-strategy", choices=["policy", "value"], default="policy")
    parser.add_argument("--opponent-time", type=float, default=0.05)
    parser.add_argument("--openings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--protocol", choices=["chessbench", "rules"], default="chessbench")
    parser.add_argument("--adjudicator", help="Stockfish executable (the reference uses Stockfish 16)")
    parser.add_argument("--max-plies", type=int, default=2048)
    parser.add_argument("--bayeselo")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.protocol == "chessbench" and not args.adjudicator:
        parser.error("--protocol chessbench requires --adjudicator")
    if args.opponent_time <= 0:
        parser.error("--opponent-time must be positive")
    torch.set_num_threads(args.threads)
    cfg = MatchConfig(games=args.games, seed=args.seed, protocol=args.protocol, max_plies=args.max_plies)
    with ExitStack() as stack:
        model = ModelAgent(args.checkpoint, args.strategy, args.device)
        if args.opponent_checkpoint:
            reference = ModelAgent(args.opponent_checkpoint, args.opponent_strategy, args.device)
        elif args.opponent_uci:
            reference = UciAgent(args.opponent_uci, args.opponent_time)
            stack.callback(reference.close)
        else:
            reference = factory_agent(args.opponent_factory)
            if callable(getattr(reference, "close", None)):
                stack.callback(reference.close)
        metadata = {"candidate": model.provenance,
                    "reference": getattr(reference, "provenance", {"factory": args.opponent_factory,
                                                                    "checkpoint_provenance_verified": False})}
        adjudicate = None
        if args.adjudicator:
            engine = chess.engine.SimpleEngine.popen_uci(args.adjudicator)
            stack.callback(engine.quit)
            metadata["adjudicator"] = {"id": engine.id, "executable_sha256": sha256_file(args.adjudicator),
                                       "time_seconds": cfg.adjudication_time, "uses_default_uci_options": True}
            adjudicate = lambda board: engine.analyse(board, chess.engine.Limit(time=cfg.adjudication_time))["score"]
        result = run_match((model, reference), ("candidate", args.opponent_name), args.openings, args.out, cfg,
                           adjudicate, metadata)
        if args.bayeselo:
            if result["summary"]["incomplete_games"]:
                raise ValueError("cannot run BayesElo on an incomplete match")
            result["bayeselo"] = run_bayeselo(args.bayeselo, args.out)
            (Path(args.out) / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
