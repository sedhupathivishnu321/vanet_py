#!/usr/bin/env python
"""STEP 10: download the real Puducherry OSM driving network (spec section 12).

Writes data/osm/puducherry.graphml and data/osm/osm_provenance.json .
"""
from __future__ import annotations

import sys
from pathlib import Path

from _common import base_parser, load, env_info


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, log = load(args)
    from src.osm.download import download_puducherry_graph, provenance
    from src.utils import ExperimentLogger

    out = Path(cfg["_meta"]["repo_root"]) / cfg["project"]["outputs_dir"]
    with ExperimentLogger(out / "logs", "download_osm", dict(cfg),
                          env_info()) as elog:
        try:
            G = download_puducherry_graph(cfg, logger=log)
        except Exception as exc:
            log.error(f"OSM download failed unexpectedly: {exc}")
            return 1
        prov = provenance()
        elog.add_metrics(n_nodes=G.number_of_nodes(), n_edges=G.number_of_edges(),
                         method=prov.get("method"), synthetic=prov.get("synthetic"),
                         n_fallbacks=len(prov.get("fallbacks", [])))
        elog.add_artifact(Path(cfg["_meta"]["repo_root"]) / cfg["osm"]["graphml"])
    if prov.get("synthetic"):
        log.warning("OSM network is the OFFLINE SYNTHETIC fallback "
                    "(Overpass unreachable). Downstream stages will run; the "
                    "report will state the network was synthetic. To use the "
                    "real network, re-run scripts/download_osm.py with Overpass "
                    "access or drop a real data/osm/puducherry.graphml in place.")
    else:
        log.info(f"OSM network ready via {prov.get('method')}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
