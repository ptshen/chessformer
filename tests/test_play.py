import chess
import pytest
import torch

from chessformer.model import Chessformer, ModelConfig
from chessformer.play import choose_move


def test_policy_returns_legal_move():
    model = Chessformer(ModelConfig(d_model=8, layers=1, heads=2, d_ff=8, value_dim=2, value_hidden=4)).eval()
    board = chess.Board()
    assert choose_move(model, board) in board.legal_moves


class ConstantValue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        return {"wdl": torch.zeros(len(inputs), 3)}


def test_value_agent_resolves_mate_without_network_approximation():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    selected = choose_move(ConstantValue(), board, "value")
    board.push(selected)
    assert board.is_checkmate()
    with pytest.raises(ValueError, match="terminal"):
        choose_move(ConstantValue(), board, "value")


class MoverPerspectiveValue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        # After White's e4, the opposing pawn appears on e5 from Black's view.
        unfavorable_for_black = inputs[:, chess.E5, 6]
        return {"wdl": torch.stack((-10 * unfavorable_for_black, torch.zeros(len(inputs)), 10 * unfavorable_for_black), -1)}


def test_value_agent_negates_child_mover_perspective():
    assert choose_move(MoverPerspectiveValue(), chess.Board(), "value").uci() == "e2e4"
