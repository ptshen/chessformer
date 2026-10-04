import chess
import pytest
import torch

from chessformer.model import Chessformer, ModelConfig
from chessformer.model_reference import exercise_relative_parameters
from chessformer.play import choose_move


@pytest.mark.parametrize("position", ("absolute", "relative_bias", "shaw"))
@pytest.mark.parametrize("implementation", ("dense", "factorized"))
def test_selected_heads_preserve_logits_attention_and_input_gradients(position, implementation):
    torch.manual_seed(41)
    cfg = ModelConfig(d_model=8, attention_dim=12, layers=2, heads=3, d_ff=11,
                      value_dim=2, value_hidden=4, value_bins=8, deepnorm="trainer",
                      position=position, attention_impl=implementation)
    model = Chessformer(cfg).eval()
    exercise_relative_parameters(model)
    inputs = torch.randn(2, 64, 112, requires_grad=True)
    expected = model(inputs, return_attention=True)
    for name in ("policy", "soft_policy", "wdl", "q", "st"):
        actual = model(inputs, heads=(name,), return_attention=True)
        assert set(actual) == {name, "attention"}
        torch.testing.assert_close(actual, {key: expected[key] for key in actual}, rtol=0, atol=0)
    without_maps = model(inputs)
    torch.testing.assert_close(model(inputs, heads=("policy", "wdl")),
                               {key: without_maps[key] for key in ("policy", "wdl")}, rtol=0, atol=0)
    combined = model(inputs, heads=("policy", "wdl"), return_attention=True)
    actual_loss = combined["policy"].square().mean() + combined["wdl"].square().mean()
    expected_loss = expected["policy"].square().mean() + expected["wdl"].square().mean()
    actual_grad = torch.autograd.grad(actual_loss, inputs)[0]
    expected_grad = torch.autograd.grad(expected_loss, inputs)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_selected_heads_skip_unused_modules_and_reject_invalid_requests():
    model = Chessformer(ModelConfig(d_model=8, layers=1, heads=2, d_ff=8, value_dim=2, value_hidden=4)).eval()
    calls = []
    names = ("policy_embedding", "policy", "soft_policy", "result", "q", "st")
    handles = [getattr(model, name).register_forward_hook(lambda m, i, o, name=name: calls.append(name))
               for name in names]
    inputs = torch.zeros(1, 64, 112)
    model(inputs, heads=("policy",))
    assert calls == ["policy_embedding", "policy"]
    calls.clear()
    model(inputs, heads=("wdl",))
    assert calls == ["result"]
    for handle in handles:
        handle.remove()
    for invalid in ((), ("unknown",), "policy"):
        with pytest.raises(ValueError, match="heads must be"):
            model(inputs, heads=invalid)


@pytest.mark.parametrize("encoding", ("cf112-v1", "lc0-training-v1:4"))
def test_selective_move_selection_matches_all_heads_for_both_strategies(encoding):
    torch.manual_seed(19)
    model = Chessformer(ModelConfig(d_model=8, layers=2, heads=2, d_ff=8, value_dim=2, value_hidden=4)).eval()
    model.input_encoding = encoding
    exercise_relative_parameters(model)
    opening = chess.Board()
    for move in "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6".split():
        opening.push_uci(move)
    boards = [opening, chess.Board(),
              chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 17 1"),
              chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
              chess.Board("7k/P7/8/8/8/8/8/K7 w - - 0 1"),
              chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")]
    for board in boards:
        for strategy in ("policy", "value"):
            expected = choose_move(model, board, strategy, batch_size=5, select_heads=False)
            actual = choose_move(model, board, strategy, batch_size=5)
            assert actual == expected and actual in board.legal_moves
    with pytest.raises(ValueError, match="batch_size must be positive"):
        choose_move(model, chess.Board(), "value", batch_size=0)


@pytest.mark.parametrize("implementation", ("dense", "factorized"))
def test_registered_matrix_flops_match_independent_shape_count(implementation):
    from chessformer.inference_benchmark import arithmetic_count
    cfg = ModelConfig(d_model=8, attention_dim=12, layers=2, heads=3, d_ff=11,
                      value_dim=2, value_hidden=4, value_bins=8, attention_impl=implementation)
    # Count multiply-accumulates directly from matrix dimensions; omit bias,
    # nonlinear, normalization, indexing, and elementwise relative cross terms.
    b, n, d, a, f, v, h = 3, 64, 8, 12, 11, 2, 4
    embedding = b * n * 112 * d
    projections = b * n * (4 * d * a + 2 * d * f)
    base_attention = 2 * b * n * n * a
    relative = 3 * b * n * (n if implementation == "dense" else 225) * a
    trunk = embedding + cfg.layers * (projections + base_attention + relative)
    policy_embedding = b * n * d * d
    policy = 2 * b * n * d * d + b * n * n * d + b * n * d * 4
    value_embedding = b * n * d * v + b * n * v * h
    result = value_embedding + b * h * 3
    auxiliary = value_embedding + b * h * (2 + cfg.value_bins)
    expected = {None: trunk + policy_embedding + 2 * policy + result + 2 * auxiliary,
                ("policy",): trunk + policy_embedding + policy, ("wdl",): trunk + result}
    with torch.device("meta"):
        model = Chessformer(cfg).eval()
        inputs = torch.empty(b, n, 112)
        for heads, macs in expected.items():
            assert arithmetic_count(model, inputs, heads)["registered_flops"] == 2 * macs
