from dataclasses import replace

import numpy as np
import pytest
import torch

from chessformer.model import Chessformer, ModelConfig, keras_variance_scaling_
from chessformer.model_reference import exercise_relative_parameters, flattened, numpy_forward, source_maps


def test_trainer_initializer_variance_truncation_and_reused_seed():
    torch.set_num_threads(1)
    beta = (8 * 15) ** -0.25
    sample = torch.empty(512, 768, dtype=torch.float64)
    keras_variance_scaling_(sample, beta, seed=42)
    requested_variance = beta / ((512 + 768) / 2)
    assert sample.var(unbiased=False).item() == pytest.approx(requested_variance, rel=0.015)
    assert sample.abs().max() <= 2 * np.sqrt(requested_variance) / 0.87962566103423978
    repeated = torch.empty_like(sample)
    keras_variance_scaling_(repeated, beta, seed=42)
    torch.testing.assert_close(sample, repeated, atol=0, rtol=0)
    model = Chessformer(ModelConfig(d_model=8, layers=2, heads=2, d_ff=12, deepnorm="trainer"))
    torch.testing.assert_close(model.blocks[0].attn.proj.weight, model.blocks[1].attn.proj.weight, atol=0, rtol=0)
    assert not torch.equal(model.blocks[0].attn.qkv.weight[:8], model.blocks[1].attn.qkv.weight[:8])


def test_independent_attention_width_with_all_heads_and_gradients():
    torch.set_num_threads(1)
    cfg = ModelConfig(d_model=10, attention_dim=12, layers=2, heads=3, d_ff=15, deepnorm="trainer")
    model = Chessformer(cfg)
    x = torch.randn(2, 64, 112)
    output = flattened(model(x, return_attention=True))
    assert output["attention"].shape == (2, 2, 3, 64, 64)
    sum(value.square().mean() for value in output.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=10, heads=3, attention_dim=11)
    with torch.device("meta"):
        candidate = Chessformer(replace(cfg, d_model=1024, attention_dim=1536, heads=32))
    assert candidate.blocks[0].attn.qkv.weight.shape == (4608, 1024)


def test_trainer_forward_and_input_gradients_against_numpy_source_equations():
    torch.set_num_threads(1)
    torch.manual_seed(21)
    try:
        maps = source_maps()
    except FileNotFoundError:
        pytest.skip("optional pinned source files unavailable; run python3 -m chessformer.reference --fetch")
    cfg = ModelConfig(d_model=8, attention_dim=12, layers=2, heads=2, d_ff=13,
                      deepnorm="trainer", shaw_cross_term=False, value_dim=2, value_hidden=5, value_bins=4)
    model = Chessformer(cfg).double()
    exercise_relative_parameters(model)
    inputs = torch.randn(1, 64, 112, dtype=torch.float64, requires_grad=True)
    actual = flattened(model(inputs, return_attention=True))
    expected = numpy_forward(model, inputs.detach().numpy(), maps)
    for key in actual:
        np.testing.assert_allclose(actual[key].detach().numpy(), expected[key], atol=1e-11, rtol=1e-11)
    # Numerical derivative of the independently implemented full model, including
    # policy promotions and value outputs, checks more than attention alone.
    def objective(values):
        return (values["policy"][0, 1800] + values["soft_policy"][0, 1840] +
                values["wdl"][0, 1] + values["q.value"][0] + values["st.category"][0, 2])
    objective(actual).backward()
    array = inputs.detach().numpy().copy()
    epsilon = 1e-5
    for square, channel in ((0, 0), (15, 9), (63, 111)):
        plus, minus = array.copy(), array.copy()
        plus[0, square, channel] += epsilon
        minus[0, square, channel] -= epsilon
        difference = (objective(numpy_forward(model, plus, maps)) - objective(numpy_forward(model, minus, maps))) / (2 * epsilon)
        assert inputs.grad[0, square, channel].item() == pytest.approx(difference, rel=2e-5, abs=1e-8)
