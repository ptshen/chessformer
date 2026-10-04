"""Legal policy and one-ply value agents, preserving history and mover perspective."""

import argparse
import json

import chess
import numpy as np
import torch

from .encoding import ENCODING, encode_board, legal_moves_by_id
from .model import Chessformer, ModelConfig
from .native import INPUT_FORMATS, NATIVE_PREFIX
from .native_encoding import encode_native_board, native_legal_moves_by_id


def encoding_functions(encoding):
    if encoding == ENCODING:
        return encode_board, legal_moves_by_id
    if isinstance(encoding, str) and encoding.startswith(NATIVE_PREFIX):
        suffix = encoding[len(NATIVE_PREFIX):]
        if suffix.isdigit() and int(suffix) in INPUT_FORMATS:
            input_format = int(suffix)
            return (lambda board: encode_native_board(board, input_format),
                    lambda board: native_legal_moves_by_id(board, input_format))
    raise ValueError("checkpoint uses an incompatible board/action encoding")


def load_model(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=False)
    encoding_functions(state.get("encoding"))
    model = Chessformer(ModelConfig(**state["model_config"]))
    model.load_state_dict(state["model"])
    model.input_encoding = state["encoding"]
    return model.to(device).eval()


@torch.no_grad()
def choose_move(model, board, strategy="policy", batch_size=64, *, select_heads=True):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    # Other compatible modules may implement only forward(inputs).
    select_heads = select_heads and isinstance(model, Chessformer)
    if board.is_game_over():
        raise ValueError("cannot select a move in a terminal position")
    device = next(model.parameters()).device
    encode, legal_map = encoding_functions(getattr(model, "input_encoding", ENCODING))
    legal = legal_map(board)
    if strategy == "policy":
        inputs = torch.from_numpy(encode(board)).unsqueeze(0).to(device)
        logits = model(inputs, heads=("policy",))["policy"][0] if select_heads else model(inputs)["policy"][0]
        return legal[max(legal, key=lambda action: float(logits[action]))]
    if strategy != "value":
        raise ValueError(strategy)
    moves = list(legal.values())
    values = np.zeros(len(moves))
    pending, encoded = [], []
    for i, move in enumerate(moves):
        child = board.copy(stack=True)
        child.push(move)
        outcome = child.outcome()
        if outcome is not None:
            values[i] = 0 if outcome.winner is None else 1 if outcome.winner == board.turn else -1
        else:
            pending.append(i)
            encoded.append(encode(child))
    for start in range(0, len(pending), batch_size):
        batch = torch.from_numpy(np.stack(encoded[start:start + batch_size])).to(device)
        # Result WDL is the main value head; q/st remain auxiliary outputs.
        output = model(batch, heads=("wdl",)) if select_heads else model(batch)
        wdl = output["wdl"].float().softmax(-1)
        child_value = (wdl[:, 0] - wdl[:, 2]).cpu().numpy()
        values[pending[start:start + batch_size]] = -child_value
    return moves[int(values.argmax())]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    parser.add_argument("--moves", nargs="*", default=[])
    parser.add_argument("--strategy", choices=["policy", "value"], default="policy")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    board = chess.Board(args.fen)
    for move in args.moves:
        board.push_uci(move)
    selected = choose_move(load_model(args.checkpoint, args.device), board, args.strategy)
    print(json.dumps({"fen": board.fen(), "strategy": args.strategy, "move": selected.uci()}))
