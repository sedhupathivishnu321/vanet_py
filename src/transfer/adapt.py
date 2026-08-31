"""Transfer-learning strategies (spec sections 10-11).

    none               - target model trained from random init (control)
    frozen_encoder     - Strategy A: load source encoder, freeze it, fine-tune
                         the AoI/temporal/adapter/head on the target
    full_finetune      - Strategy B: load the whole source model, fine-tune end
                         to end at a low learning rate
    frozen_encoder_da  - Strategy A plus an explicit domain-alignment loss
                         (MMD or CORAL) between source and target latents
"""
from __future__ import annotations

import numpy as np
import torch

from src.models.losses import domain_loss

TRANSFER_STRATEGIES = ["none", "frozen_encoder", "full_finetune",
                       "frozen_encoder_da"]


def load_encoder_from_source(target_model: torch.nn.Module,
                             source_state: dict,
                             strict: bool = False) -> list[str]:
    """Copy every parameter whose name + shape matches. Returns the list of
    successfully transferred parameter names."""
    tgt_state = target_model.state_dict()
    transferred = []
    for k, v in source_state.items():
        if k in tgt_state and tgt_state[k].shape == v.shape:
            tgt_state[k] = v.clone()
            transferred.append(k)
    target_model.load_state_dict(tgt_state, strict=strict)
    return transferred


def apply_transfer_strategy(target_model, source_state, strategy: str):
    """Mutates `target_model` in place; returns (param_groups, meta)."""
    meta = {"strategy": strategy, "transferred_params": 0,
            "frozen_encoder": False, "domain_adaptation": False}
    if strategy == "none":
        return [p for p in target_model.parameters() if p.requires_grad], meta

    transferred = load_encoder_from_source(target_model, source_state)
    meta["transferred_params"] = len(transferred)

    if strategy in ("frozen_encoder", "frozen_encoder_da"):
        if hasattr(target_model, "freeze_encoder"):
            target_model.freeze_encoder(True)
            meta["frozen_encoder"] = True
            groups = list(target_model.adapt_parameters())
        else:  # non-proposed model without an explicit encoder split
            groups = [p for p in target_model.parameters() if p.requires_grad]
        if strategy == "frozen_encoder_da":
            meta["domain_adaptation"] = True
        return groups, meta

    if strategy == "full_finetune":
        for p in target_model.parameters():
            p.requires_grad = True
        return [p for p in target_model.parameters()], meta

    raise KeyError(strategy)


def make_domain_aux_loss(source_model, source_X: np.ndarray, adj_source_t,
                         kind: str, weight: float, device: str, seed: int = 0):
    """Return an ``aux(model, xb, ab)`` closure adding lambda_DA * domain_loss
    between the target model's latent(xb) and the (frozen) source model's
    latent on a rotating batch of source windows.  Returns ``None`` if either
    model lacks a ``latent`` method (e.g. non-proposed baselines)."""
    if not hasattr(source_model, "latent"):
        return None
    src_x = torch.from_numpy(np.asarray(source_X, np.float32))
    rng = np.random.default_rng(seed)
    source_model.to(device).eval()

    def aux(model, xb, ab):
        if not hasattr(model, "latent"):
            return xb.new_zeros(())
        n = min(xb.shape[0], src_x.shape[0])
        pick = rng.integers(0, src_x.shape[0], size=n)
        sx = src_x[pick].to(device)
        with torch.no_grad():
            zs = source_model.latent(sx, adj_source_t)
        zt = model.latent(xb[:n], _adj_of(model), ab[:n] if ab is not None else None)
        return weight * domain_loss(zs, zt, kind)

    # adjacency for the target model is captured lazily via attribute set by caller
    def _adj_of(model):
        return getattr(model, "_da_adj_t", None)

    return aux
