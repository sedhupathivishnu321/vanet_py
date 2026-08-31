"""Model registry.

`build_model(name, ...)` returns an ``nn.Module`` for every torch baseline and
the proposed model.  Non-parametric baselines (historical average, linear
regression, random forest) live in :mod:`src.models.baselines` and expose a
plain ``fit``/``predict`` numpy API instead.
"""
from __future__ import annotations

TORCH_MODELS = {
    "lstm", "gru", "gcn", "gat", "stgnn", "transformer", "proposed",
}
NUMPY_BASELINES = {"historical_average", "linear_regression", "random_forest"}
ALL_MODELS = sorted(TORCH_MODELS | NUMPY_BASELINES)


def build_model(name: str, in_dim: int, num_nodes: int, horizon: int,
                cfg, adj=None):
    """Instantiate a torch model by name. `cfg` is the global Config."""
    name = name.lower()
    tr = cfg["training"]
    hidden = int(tr["hidden_dim"])
    dropout = float(tr["dropout"])
    from .rnn import LSTMForecaster, GRUForecaster
    from .gnn import GCNForecaster, GATForecaster, STGNNForecaster
    from .transformer import TemporalTransformerForecaster
    from .proposed import AoITransferableSTGNN

    if name == "lstm":
        return LSTMForecaster(in_dim, num_nodes, hidden, horizon, dropout)
    if name == "gru":
        return GRUForecaster(in_dim, num_nodes, hidden, horizon, dropout)
    if name == "gcn":
        return GCNForecaster(in_dim, num_nodes, hidden, horizon, dropout)
    if name == "gat":
        return GATForecaster(in_dim, num_nodes, hidden, horizon, dropout,
                             heads=int(cfg["proposed_model"]["gat_heads"]))
    if name == "stgnn":
        return STGNNForecaster(in_dim, num_nodes, hidden, horizon, dropout)
    if name == "transformer":
        return TemporalTransformerForecaster(in_dim, num_nodes, hidden, horizon,
                                             dropout)
    if name == "proposed":
        pm = cfg["proposed_model"]
        return AoITransferableSTGNN(
            in_dim, num_nodes, hidden, horizon, dropout,
            spatial=pm["spatial_encoder"], temporal=pm["temporal_encoder"],
            heads=int(pm["gat_heads"]), aoi_attention=bool(pm["aoi_attention"]),
            aoi_beta=float(pm["aoi_beta"]))
    raise KeyError(f"unknown torch model '{name}'")


__all__ = ["build_model", "TORCH_MODELS", "NUMPY_BASELINES", "ALL_MODELS"]
