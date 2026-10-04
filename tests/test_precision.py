import torch

from chessformer.data import record_arrays
from chessformer.encoding import ENCODING
from chessformer.losses import LossConfig, multitask_loss
from chessformer.model import Chessformer, ModelConfig


def test_bfloat16_multitask_step_is_finite():
    torch.set_num_threads(1)
    model = Chessformer(ModelConfig(d_model=16, layers=1, heads=2, d_ff=16, value_dim=2,
                                    value_hidden=8, attention_impl="factorized", value_bins=8))
    record = {"encoding": ENCODING, "game_id": "precision", "policy": {"e2e4": 3, "d2d4": 1},
              "wdl": [0, 1, 0], "q": 0.2, "st": 0.1}
    batch = {k: torch.as_tensor(v).unsqueeze(0) for k, v in record_arrays(record).items()}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(batch["inputs"])
    loss, _ = multitask_loss(output, batch, LossConfig(category=0.1))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
