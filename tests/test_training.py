from dataclasses import replace
import json

import chess
import pytest
import torch

from chessformer.data import PositionDataset, assert_disjoint, record_arrays
from chessformer.encoding import ENCODING, encode_board
from chessformer.losses import LossConfig, multitask_loss, policy_loss
from chessformer.model import Chessformer, ModelConfig
from chessformer.train import TrainConfig, run_training


@pytest.fixture(autouse=True)
def small_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def record(game_id="a", moves=None, **kwargs):
    return {"encoding": ENCODING, "game_id": game_id, "moves": moves or [],
            "policy": {"e2e4": 1}, "wdl": [0, 1, 0], "q": 0.2, "st": -0.1, **kwargs}


def test_rejects_missing_aux_and_illegal_policy():
    with pytest.raises(ValueError, match="missing q"):
        record_arrays(record(q=None))
    assert not record_arrays(record(q=None), require_aux=False)["q_valid"]
    with pytest.raises(ValueError, match="invalid policy"):
        record_arrays(record(policy={"e2e5": 1}))


def test_legal_zero_mass_moves_stay_in_denominator_and_illegal_gradient_zero():
    logits = torch.tensor([[1., 1., 100.]], requires_grad=True)
    ce, _ = policy_loss(logits, torch.tensor([[1., 0., 0.]]), torch.tensor([[True, True, False]]))
    assert float(ce.detach()) == pytest.approx(0.69314718)
    ce.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[-0.5, 0.5, 0.]]))


@pytest.mark.parametrize("position", ["absolute", "relative_bias", "shaw"])
def test_all_heads_train_and_attention_shapes(position):
    cfg = ModelConfig(d_model=16, layers=2, heads=2, d_ff=24, value_dim=4, value_hidden=8,
                      position=position, value_bins=8)
    model = Chessformer(cfg)
    arrays = record_arrays(record())
    batch = {k: torch.as_tensor(v).unsqueeze(0) for k, v in arrays.items()}
    output = model(batch["inputs"], return_attention=True)
    assert output["policy"].shape == (1, 1858)
    assert output["attention"].shape == (1, 2, 2, 64, 64)
    torch.testing.assert_close(output["attention"].sum(-1), torch.ones(1, 2, 2, 64))
    loss, _ = multitask_loss(output, batch, LossConfig(category=0.1))
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_error_loss_does_not_backpropagate_into_reward_prediction():
    model = Chessformer(ModelConfig(d_model=8, layers=1, heads=2, d_ff=8, value_dim=2, value_hidden=4))
    batch = {k: torch.as_tensor(v).unsqueeze(0) for k, v in record_arrays(record()).items()}
    output = model(batch["inputs"])
    output["q"]["value"].retain_grad()
    _, metrics = multitask_loss(output, batch)
    metrics["q_error_mse"].backward()
    assert output["q"]["value"].grad is None


def make_datasets(tmp_path):
    trainpath, valpath = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    trainpath.write_text(json.dumps(record()) + "\n" + json.dumps(record("b", ["g1f3"], policy={"d7d5": 1}, q=None)) + "\n")
    valpath.write_text(json.dumps(record("c", ["e2e4"], policy={"e7e5": 1})) + "\n")
    return PositionDataset(trainpath, require_aux=False), PositionDataset(valpath)


def test_overlap_is_rejected(tmp_path):
    train, val = make_datasets(tmp_path)
    assert_disjoint(train, val)
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(train, train)
    val.game_ids = {"unrelated"}
    val.samples = train.samples
    with pytest.raises(ValueError, match="identical encoded"):
        assert_disjoint(train, val)


def test_resume_matches_uninterrupted_training_and_swa(tmp_path):
    train, val = make_datasets(tmp_path)
    model = ModelConfig(d_model=8, layers=1, heads=2, d_ff=8, value_dim=2, value_hidden=4)
    cfg = TrainConfig(steps=4, batch_size=3, microbatch_size=2, swa_start=2, swa_every=1,
                      checkpoint_every=2, eval_every=4, lr_milestones=(2,), lr_values=(1e-4,))
    run_training(model, cfg, LossConfig(), train, val, tmp_path / "full")
    run_training(model, cfg, LossConfig(), train, val, tmp_path / "split", stop_after=2)
    run_training(model, cfg, LossConfig(), train, val, tmp_path / "split", resume=True)
    full = torch.load(tmp_path / "full/checkpoint.pt", weights_only=False)
    split = torch.load(tmp_path / "split/checkpoint.pt", weights_only=False)
    for key in ("model", "swa"):
        for name in full[key]:
            torch.testing.assert_close(full[key][name], split[key][name], rtol=0, atol=0)
    assert full["step"] == split["step"] == 4
    with pytest.raises(ValueError, match="fingerprint"):
        run_training(model, replace(cfg, lr=1e-3), LossConfig(), train, val, tmp_path / "split", resume=True)


def test_accumulation_matches_full_batch_even_with_missing_aux(tmp_path):
    train, val = make_datasets(tmp_path)
    model = ModelConfig(d_model=8, layers=1, heads=2, d_ff=8, value_dim=2, value_hidden=4)
    cfg = TrainConfig(steps=1, batch_size=5, microbatch_size=2)
    run_training(model, cfg, LossConfig(), train, val, tmp_path / "micro")
    run_training(model, replace(cfg, microbatch_size=5), LossConfig(), train, val, tmp_path / "whole")
    micro = torch.load(tmp_path / "micro/checkpoint.pt", weights_only=False)
    whole = torch.load(tmp_path / "whole/checkpoint.pt", weights_only=False)
    # Check gradients through the optimizer's first moments. Near-zero gradients
    # can change sign with float32 reduction order; NAdam can magnify that noise
    # in redundant attention biases, so weight equality is the wrong invariant.
    for key in micro["optimizer"]["state"]:
        torch.testing.assert_close(micro["optimizer"]["state"][key]["exp_avg"],
                                   whole["optimizer"]["state"][key]["exp_avg"], atol=1e-7, rtol=2e-5)
    models = [Chessformer(model), Chessformer(model)]
    for m, state in zip(models, (micro, whole)):
        m.load_state_dict(state["model"])
    inputs = val.batch(range(len(val)), "cpu")["inputs"]
    with torch.no_grad():
        a, b = [m(inputs) for m in models]
    for key in ("policy", "soft_policy", "wdl"):
        torch.testing.assert_close(a[key].softmax(-1), b[key].softmax(-1), atol=1e-6, rtol=2e-5)
