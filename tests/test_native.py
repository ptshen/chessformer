import gzip
import io
import json
import sqlite3
import struct

import chess
import numpy as np
import pytest
import torch

from chessformer.data import assert_disjoint, load_dataset
from chessformer.losses import LossConfig
from chessformer.model import ModelConfig
from chessformer.native import decode_record, read_game, short_term_values
from chessformer.native_data import NativeDataset, build_native_dataset
from chessformer.play import choose_move, load_model
from chessformer.train import TrainConfig, run_training
from native_fixture import game_bytes, record, write_archive


def test_native_plane_bit_order_castling_ep_and_clock():
    board = chess.Board()
    board.push_uci("e2e4")
    raw = bytearray(record(board, version=7))
    raw[8276] = 1 << 4  # Explicit protocol EP bit, independent of pawn capturability.
    raw[8277] = 17
    sample = decode_record(raw)
    x = sample["inputs"]
    assert x[chess.E2, 0] == 1
    assert x[chess.E5, 6] == 1
    assert x[chess.E8, 108] == 1
    assert x[chess.A1, 104] == x[chess.H8, 105] == 1
    assert x[0, 109] == np.float32(0.17)
    assert sample["st"] == np.float32(-0.4)
    assert sample["legal"].sum() == board.legal_moves.count()
    assert (sample["policy"][sample["legal"]] == 0).sum() == board.legal_moves.count() - 2


def test_v6_reserved_float_is_never_treated_as_short_term_reward():
    raw = game_bytes(7, plies=3)
    with pytest.raises(ValueError, match="complete chronological game"):
        decode_record(raw[:8356])
    game = read_game(io.BytesIO(raw))
    expected = short_term_values([0.1, 0.3, 0.5])
    for (ply, sample), value in zip(game.samples(), expected):
        assert sample["st"] == pytest.approx(value)
    assert game.version == 6


def test_future_reward_ema_sign_and_terminal_tail():
    assert short_term_values([0.75])[0] == 0.75
    np.testing.assert_allclose(short_term_values([0.2, -0.8]), [0.7, -0.8], atol=1e-7)
    with pytest.raises(ValueError, match="nonempty"):
        short_term_values([])


def test_game_identity_ignores_rescoring_but_preserves_positions():
    original = bytearray(game_bytes(1))
    rescored = original.copy()
    struct.pack_into("<f", rescored, 8280, 0.55)
    a, b = [read_game(io.BytesIO(raw)) for raw in (original, rescored)]
    assert a.game_id == b.game_id
    assert a.content_sha256 != b.content_sha256
    assert a.game_id != read_game(io.BytesIO(game_bytes(2))).game_id


@pytest.mark.parametrize("mutation,match", [
    (lambda raw: raw[:-1], "truncated"),
    (lambda raw: raw + b"x", "truncated"),
    (lambda raw: bytes([9]) + raw[1:], "unsupported record version"),
    (lambda raw: raw[:8356] + raw[:8356], "non-alternating"),
])
def test_native_rejects_invalid_game_streams(mutation, match):
    with pytest.raises(ValueError, match=match):
        read_game(io.BytesIO(mutation(game_bytes())))


def test_native_limits_and_format_mismatch():
    with pytest.raises(ValueError, match="max_plies"):
        read_game(io.BytesIO(game_bytes()), max_plies=2)
    with pytest.raises(ValueError, match="expected input format"):
        read_game(io.BytesIO(game_bytes()), expected_input_format=1)
    with pytest.raises(ValueError, match="max_plies"):
        read_game(io.BytesIO(game_bytes()), max_plies=0)


def test_native_invalid_targets_fail_instead_of_creating_neutral_labels():
    raw = bytearray(record(chess.Board(), version=7))
    struct.pack_into("<f", raw, 8280, float("nan"))
    with pytest.raises(ValueError, match="distribution"):
        decode_record(raw)
    raw = bytearray(record(chess.Board(), version=7))
    struct.pack_into("<f", raw, 8, -0.5)
    with pytest.raises(ValueError, match="sentinel"):
        decode_record(raw)


def dataset(tmp_path):
    path = tmp_path / "corpus.sqlite"
    archive = write_archive(tmp_path / "fixtures.tar")
    manifest = build_native_dataset([archive], path, input_format=4, validation_fraction=0.3)
    return path, archive, manifest


def test_sparse_disk_round_trip_and_disjoint_splits(tmp_path):
    path, _, manifest = dataset(tmp_path)
    train, validation = NativeDataset(path, "train"), NativeDataset(path, "validation")
    try:
        assert_disjoint(train, validation)
        assert train.encoding == validation.encoding == "lc0-training-v1:4"
        assert manifest["duplicate_inputs"] > 0
        assert not hasattr(train, "samples")
        for split in (train, validation):
            a, b = split.batch([0, len(split) - 1, 0], "cpu"), split.batch([0], "cpu")
            for key in a:
                torch.testing.assert_close(a[key][0], b[key][0], atol=0, rtol=0)
                torch.testing.assert_close(a[key][2], b[key][0], atol=0, rtol=0)
            assert torch.all(a["policy"][~a["legal"]] == 0)
            torch.testing.assert_close(a["policy"].sum(-1), torch.ones(3))
        assert train.fingerprint != validation.fingerprint
        with pytest.raises(ValueError, match="identical encoded"):
            assert_disjoint(train, train)
    finally:
        train.close()
        validation.close()


def test_native_detects_cross_database_overlap(tmp_path):
    path, archive, _ = dataset(tmp_path)
    second = tmp_path / "other.sqlite"
    build_native_dataset([archive], second, input_format=4, validation_fraction=0.3)
    a, b = NativeDataset(path, "train"), NativeDataset(second, "train")
    try:
        with pytest.raises(ValueError, match="identical encoded"):
            assert_disjoint(a, b)
    finally:
        a.close()
        b.close()


def test_native_training_and_exact_resume_with_all_aux_targets(tmp_path):
    torch.set_num_threads(1)
    path, _, _ = dataset(tmp_path)
    train, validation = load_dataset(path, "train"), load_dataset(path, "validation")
    model = ModelConfig(d_model=8, attention_dim=12, layers=1, heads=2, d_ff=8,
                        deepnorm="trainer", shaw_cross_term=False, value_dim=2, value_hidden=4, value_bins=8)
    cfg = TrainConfig(steps=3, batch_size=3, microbatch_size=2, swa_start=1, swa_every=1)
    try:
        run_training(model, cfg, LossConfig(category=0.1), train, validation, tmp_path / "full")
        run_training(model, cfg, LossConfig(category=0.1), train, validation, tmp_path / "split", stop_after=1)
        run_training(model, cfg, LossConfig(category=0.1), train, validation, tmp_path / "split", resume=True)
        a = torch.load(tmp_path / "full/model.pt", weights_only=False)
        b = torch.load(tmp_path / "split/model.pt", weights_only=False)
        assert a["encoding"] == "lc0-training-v1:4"
        for name in a["model"]:
            torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
        # Native checkpoints must select native inference conventions.
        loaded = load_model(tmp_path / "full/model.pt")
        assert loaded.input_encoding == "lc0-training-v1:4"
        assert loaded.cfg.deepnorm == "trainer" and loaded.cfg.attention_width == 12
        assert choose_move(loaded, chess.Board()) in chess.Board().legal_moves
        run = json.loads((tmp_path / "full/manifest.json").read_text())
        assert run["available_aux"] == {"q": len(train), "st": len(train)}
    finally:
        train.close()
        validation.close()


def test_failed_import_is_not_published_and_good_outputs_are_not_overwritten(tmp_path):
    path, archive, _ = dataset(tmp_path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        build_native_dataset([archive], path, input_format=4)
    assert path.read_bytes() == before
    broken = tmp_path / "truncated.gz"
    broken.write_bytes(gzip.compress(game_bytes()[:-1]))
    out = tmp_path / "bad.sqlite"
    with pytest.raises(ValueError, match="truncated"):
        build_native_dataset([broken], out, input_format=4)
    assert not out.exists()


def test_old_database_cannot_bypass_chronology_validation(tmp_path):
    path, _, manifest = dataset(tmp_path)
    assert manifest["chronology_validation"] and not manifest["chunk_endpoints_verified"]
    manifest["schema"] = "chessformer-native-sqlite-v1"
    with sqlite3.connect(path) as db:
        db.execute("UPDATE metadata SET value=? WHERE key='manifest'", (json.dumps(manifest),))
    with pytest.raises(ValueError, match="reimport complete chunks"):
        NativeDataset(path, "train")
