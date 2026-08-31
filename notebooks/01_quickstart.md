# Quickstart notebook (Markdown form)

This project ships pipeline **scripts** rather than heavy notebooks so results
are reproducible from the command line. Use this file as a copy-paste session.

```python
import sys; sys.path.insert(0, "..")
from src.utils import load_config, set_seed
cfg = load_config(profile="quick")
set_seed(42)

# 1. Source dataset --------------------------------------------------------
from src.data.source_prep import prepare_source_domain
prep = prepare_source_domain(cfg)
print(prep["dataset_name"], "synthetic:", prep["is_synthetic"],
      "nodes:", prep["num_nodes"])

# 2. Build + forward the proposed model ---------------------------------
import torch
from src.models import build_model
model = build_model("proposed", prep["in_dim"], prep["num_nodes"],
                    prep["horizon"], cfg)
adj_t = torch.from_numpy(prep["adj_norm"])
xb = prep["torch"]["train"]["x"][:4]
print("prediction shape:", model(xb, adj_t).shape)   # (4, H, N, 1)

# 3. Target scenario (OSM corridor -> IDM traffic -> VANET) -------------
from src.experiments import run_scenario
sc = run_scenario(cfg, "corridor_1", seed=42)
print("backend:", sc["backend"],
      "mean AoI:", sc["partial_obs"].aoi_stats["mean"])

# 4. Closed-loop control ------------------------------------------------
from src.sumo import run_target_simulation
from src.control import build_controller
from src.experiments import load_corridor
out = run_target_simulation(cfg, load_corridor(cfg, "corridor_1"), "MEDIUM", 42,
                            controller=build_controller("max_pressure", cfg))
print(out.metrics)
```

To run everything end to end instead:

```bash
python run_all.py --quick
```
