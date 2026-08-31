import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models import build_model
from src.transfer import apply_transfer_strategy, load_encoder_from_source
from src.vanet import CommunicationChannel, build_partial_observation
from src.sumo.idm_sim import IDMSimulator
from src.preprocessing.target_features import build_target_tensors


def test_encoder_partial_transfer_matching_shapes(cfg):
    src = build_model("proposed", 8, 20, 12, cfg)      # 20-node source
    tgt = build_model("proposed", 8, 6, 12, cfg)       # 6-node target
    moved = load_encoder_from_source(tgt, src.state_dict())
    # hidden-space encoder params (shape independent of N) should transfer
    assert any("enc1" in m or "input_proj" in m for m in moved)


@pytest.mark.parametrize("strat", ["none", "frozen_encoder", "full_finetune",
                                   "frozen_encoder_da"])
def test_apply_transfer_strategy(cfg, strat):
    src = build_model("proposed", 8, 8, 12, cfg)
    tgt = build_model("proposed", 8, 8, 12, cfg)
    groups, meta = apply_transfer_strategy(tgt, src.state_dict(), strat)
    assert meta["strategy"] == strat
    assert len(list(groups)) > 0
    if strat == "frozen_encoder":
        assert meta["frozen_encoder"] is True


def test_full_target_tensor_assembly(cfg, tiny_corridor):
    out = IDMSimulator(cfg, tiny_corridor, "LOW", seed=0).run()
    ch = CommunicationChannel(0.5, 0.9, 100, seed=0)
    po = build_partial_observation(out, ch, cfg)
    assert po.observed_state.shape == po.ground_truth_state.shape
    assert po.mask.min() >= 0 and po.mask.max() <= 1
    tn = build_target_tensors(po, cfg, seed=0)
    assert tn["in_dim"] == 8
    # chronological: test window indices strictly after val
    if len(tn["test"]["t_index"]) and len(tn["val"]["t_index"]):
        assert tn["test"]["t_index"].min() >= tn["val"]["t_index"].max()
