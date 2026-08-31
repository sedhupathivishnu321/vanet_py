#!/usr/bin/env python
"""STEP 11-13: geocode the endpoints and build the two study corridors
(spec section 13).

Writes:
    data/osm/corridor_routes.geojson
    data/osm/corridor_1_ml_graph.json
    data/osm/corridor_2_ml_graph.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from _common import base_parser, load, env_info


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, log = load(args)
    from src.osm.download import load_graph, download_puducherry_graph
    from src.osm.corridors import build_corridors
    from src.utils import ExperimentLogger

    out = Path(cfg["_meta"]["repo_root"]) / cfg["project"]["outputs_dir"]
    with ExperimentLogger(out / "logs", "build_routes", dict(cfg),
                          env_info()) as elog:
        try:
            G = load_graph(cfg)
        except FileNotFoundError:
            log.warning("puducherry.graphml missing - building it now "
                        "(scripts/download_osm.py was skipped or failed)")
            G = download_puducherry_graph(cfg, logger=log)
        results = build_corridors(G, cfg, logger=log)
        for r in results:
            elog.add_metrics(**{f"{r.id}_length_m": r.length_m,
                                f"{r.id}_intersections": r.n_intersections,
                                f"{r.id}_segments": r.n_segments,
                                f"{r.id}_fallback_geocode": r.used_fallback_geocode})
        elog.add_artifact(Path(cfg["_meta"]["repo_root"]) / cfg["osm"]["corridor_geojson"])

    log.info("corridor summary")
    for r in results:
        log.info(f"  {r.id:11s} {r.name:32s} {r.length_m:8.0f} m  "
                 f"{r.n_segments:3d} segments  {r.n_intersections:3d} intersections"
                 f"  fallback_geocode={r.used_fallback_geocode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
