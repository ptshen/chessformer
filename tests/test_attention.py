import pytest
import torch

from chessformer.model import relative_indices, shaw_attention


@pytest.mark.parametrize("implementation", ["dense", "factorized"])
@pytest.mark.parametrize("cross_term", [True, False])
def test_shaw_matches_literal_equation_and_gradients(implementation, cross_term):
    torch.manual_seed(9)
    tensors = [torch.randn(2, 2, 64, 3, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    tensors += [torch.randn(2, 225, 3, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    q, k, v, rq, rk, rv = tensors
    idx = relative_indices()
    aq, ak, av = rq[:, idx], rk[:, idx], rv[:, idx]
    if cross_term:
        scores = ((q.unsqueeze(-2) + aq) * (k.unsqueeze(-3) + ak)).sum(-1)
    else:
        scores = (q.unsqueeze(-2) * k.unsqueeze(-3) + q.unsqueeze(-2) * ak + k.unsqueeze(-3) * aq).sum(-1)
    weights = (scores / 3 ** 0.5).softmax(-1)
    expected = (weights.unsqueeze(-1) * (v.unsqueeze(-3) + av)).sum(-2)
    actual, actual_weights = shaw_attention(*tensors, idx, implementation=implementation, cross_term=cross_term)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual_weights, weights, atol=1e-12, rtol=1e-12)
    probe = torch.randn_like(actual)
    expected_grad = torch.autograd.grad((expected * probe).sum(), tensors, retain_graph=True)
    actual_grad = torch.autograd.grad((actual * probe).sum(), tensors)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)


def test_displacements_are_two_dimensional_and_translation_tied():
    index = relative_indices()
    assert index[0, 1] == index[8, 9]
    assert index[0, 1] != index[7, 8]  # prevent flattened-row wraparound
    assert index[0, 8] != index[0, 1]
    assert index.unique().numel() == 225


def test_zero_relative_vectors_equal_standard_attention():
    torch.manual_seed(13)
    q, k, v = [torch.randn(1, 2, 64, 4) for _ in range(3)]
    zero = torch.zeros(2, 225, 4)
    actual, _ = shaw_attention(q, k, v, zero, zero, zero, relative_indices())
    torch.testing.assert_close(actual, torch.nn.functional.scaled_dot_product_attention(q, k, v))
