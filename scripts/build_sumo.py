#!/usr/bin/env python
"""STEP 14: build the SUMO target network from OSM (spec section 17).

If SUMO is unavailable or conversion fails, the project falls back to the
built-in IDM micro-simulator; this is logged and recorded, never hidden.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from _common import base_parser, load, env_info


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, log = load(args)
    from src.sumo.build_network import build_sumo_network
    from src.utils import ExperimentLogger

    out = Path(cfg["_meta"]["repo_root"]) / cfg["project"]["outputs_dir"]
    with ExperimentLogger(out / "logs", "build_sumo", dict(cfg), env_info()) as elog:
        status = build_sumo_network(cfg, logger=log)
        (Path(cfg["_meta"]["repo_root"]) / "data" / "sumo" / "build_status.json"
         ).write_text(json.dumps(status, indent=2))
        elog.add_metrics(**{k: v for k, v in status.items()
                            if isinstance(v, (bool, str, int, float))})

    log.info(f"SUMO available : {status['sumo_available']}  "
             f"({status.get('sumo_version')})")
    log.info(f"network built  : {status['built']}")
    log.info(f"note           : {status['note']}")
    if not status["built"]:
        log.info("=> target-domain simulation will use the IDM micro-simulator "
                 "(clearly labelled 'idm_fallback' in every artefact).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
