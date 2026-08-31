"""Retrieve the *real* Puducherry driving network from OpenStreetMap via OSMnx.

We never invent the road network.  If the OSM download or Nominatim geocoding
fails (offline Codespace, rate limiting) the failure is logged and a clearly
recorded fallback (bounding circle / hard-coded approximate coordinates from
``config.yaml``) is used so the rest of the pipeline can still run.  Every such
fallback is written into ``data/osm/osm_provenance.json``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

_PROV: dict = {"retrieved_utc": None, "method": None, "fallbacks": []}


def _osmnx():
    import osmnx as ox
    # be a good Nominatim citizen
    try:
        ox.settings.log_console = False
        ox.settings.use_cache = True
        ox.settings.requests_timeout = 60
    except Exception:
        pass
    return ox


def geocode_point(query: str, fallback_latlon, logger=None):
    """Return (lat, lon). Falls back to `fallback_latlon` on any failure."""
    try:
        ox = _osmnx()
        lat, lon = ox.geocode(query)
        if logger:
            logger.info(f"  geocoded {query!r} -> ({lat:.5f}, {lon:.5f})")
        return float(lat), float(lon), False
    except Exception as exc:  # pragma: no cover - network dependent
        if logger:
            logger.warning(f"  geocode failed for {query!r} ({exc}); using "
                           f"fallback {fallback_latlon}")
        _PROV["fallbacks"].append({"query": query, "fallback": list(fallback_latlon),
                                   "error": str(exc)})
        return float(fallback_latlon[0]), float(fallback_latlon[1]), True


def download_puducherry_graph(cfg, logger=None):
    ox = _osmnx()
    o = cfg["osm"]
    out = Path(cfg["_meta"]["repo_root"]) / o["graphml"]
    out.parent.mkdir(parents=True, exist_ok=True)
    G = None
    method = None
    try:
        G = ox.graph_from_place(o["place"], network_type=o["network_type"])
        method = f"graph_from_place({o['place']!r})"
    except Exception as exc:
        if logger:
            logger.warning(f"graph_from_place failed ({exc}); trying bounding "
                           f"circle around {o['fallback_center']}")
        _PROV["fallbacks"].append({"stage": "graph", "error": str(exc)})
        try:
            G = ox.graph_from_point(tuple(o["fallback_center"]),
                                    dist=int(o["fallback_radius_m"]),
                                    network_type=o["network_type"])
            method = (f"graph_from_point({tuple(o['fallback_center'])}, "
                      f"dist={o['fallback_radius_m']})")
        except Exception as exc2:  # pragma: no cover
            raise RuntimeError(
                "Could not retrieve the Puducherry OSM network (no internet?). "
                "Provide data/osm/puducherry.graphml manually.") from exc2

    # enrich with speeds / travel times where OSM tags allow
    try:
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
    except Exception:
        pass

    ox.save_graphml(G, out)
    _PROV["retrieved_utc"] = datetime.now(timezone.utc).isoformat()
    _PROV["method"] = method
    _PROV["n_nodes"] = G.number_of_nodes()
    _PROV["n_edges"] = G.number_of_edges()
    prov_path = out.parent / "osm_provenance.json"
    with open(prov_path, "w", encoding="utf-8") as fh:
        json.dump(_PROV, fh, indent=2)
    if logger:
        logger.info(f"  saved {out}  ({G.number_of_nodes()} nodes, "
                    f"{G.number_of_edges()} edges)  via {method}")
    return G


def load_graph(cfg):
    ox = _osmnx()
    path = Path(cfg["_meta"]["repo_root"]) / cfg["osm"]["graphml"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/download_osm.py")
    return ox.load_graphml(path)


def provenance() -> dict:
    return dict(_PROV)
