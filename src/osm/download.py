"""Retrieve the *real* Puducherry driving network from OpenStreetMap via OSMnx.

We never invent the road network when OSM is reachable.  Strategy, fastest and
most reliable first:

  1. ``graph_from_point(center, dist)`` against several Overpass mirrors with a
     short timeout (avoids the huge multi-polygon of the Puducherry UT, which
     spans Mahé / Yanam / Karaikal and triggers Overpass area-limit splitting).
  2. ``graph_from_place`` as a secondary attempt.
  3. If every mirror is unreachable, an **offline anchored fallback** graph is
     built from the corridor endpoint coordinates in ``config.yaml`` and saved
     as a valid GraphML, with ``method: offline_synthetic_fallback`` recorded in
     ``data/osm/osm_provenance.json``.  Downstream stages then run normally and
     the final report states clearly that the network was synthetic.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

_PROV: dict = {"retrieved_utc": None, "method": None, "fallbacks": []}

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
    "https://overpass.openstreetmap.ru/api",
]


def _osmnx(timeout: int = 25):
    import osmnx as ox
    try:
        ox.settings.log_console = False
        ox.settings.use_cache = True
        ox.settings.requests_timeout = timeout
        ox.settings.overpass_rate_limit = False
    except Exception:
        pass
    return ox


def _set_overpass(ox, url: str) -> None:
    for attr in ("overpass_url", "overpass_endpoint"):
        try:
            setattr(ox.settings, attr, url)
        except Exception:
            pass


def geocode_point(query: str, fallback_latlon, logger=None, offline: bool = False):
    """Return (lat, lon, used_fallback). Uses the config fallback on any failure
    or when ``offline`` is set (no Nominatim round-trip)."""
    if offline:
        return float(fallback_latlon[0]), float(fallback_latlon[1]), True
    try:
        ox = _osmnx(timeout=15)
        lat, lon = ox.geocode(query)
        if logger:
            logger.info(f"  geocoded {query!r} -> ({lat:.5f}, {lon:.5f})")
        return float(lat), float(lon), False
    except Exception as exc:  # pragma: no cover - network dependent
        if logger:
            logger.warning(f"  geocode failed for {query!r} ({exc}); "
                           f"using fallback {fallback_latlon}")
        _PROV["fallbacks"].append({"query": query, "fallback": list(fallback_latlon),
                                   "error": str(exc)[:200]})
        return float(fallback_latlon[0]), float(fallback_latlon[1]), True


# --------------------------------------------------------------------------- #
#  Offline anchored fallback network
# --------------------------------------------------------------------------- #
def _haversine(lat1, lon1, lat2, lon2):
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.asin(math.sqrt(a))


def build_offline_graph(cfg, logger=None):
    """A connected MultiDiGraph anchored on the two corridors' endpoints, dense
    enough for nearest-node snapping and shortest-path routing.  Clearly marked
    synthetic."""
    import networkx as nx

    G = nx.MultiDiGraph()
    G.graph["crs"] = "epsg:4326"
    G.graph["synthetic"] = True
    G.graph["created_utc"] = datetime.now(timezone.utc).isoformat()
    nid = [0]
    corridor_node_ids = {}

    def _add_line(pts, hwy="secondary", spd=40):
        ids = []
        for lat, lon in pts:
            G.add_node(nid[0], x=float(lon), y=float(lat), street_count=2)
            ids.append(nid[0]); nid[0] += 1
        for a, b in zip(ids, ids[1:]):
            d = _haversine(G.nodes[a]["y"], G.nodes[a]["x"],
                           G.nodes[b]["y"], G.nodes[b]["x"])
            for u, v in ((a, b), (b, a)):
                G.add_edge(u, v, osmid=f"syn{u}_{v}", length=round(d, 2),
                           highway=hwy, oneway=False, maxspeed=spd,
                           speed_kph=spd, name="synthetic corridor")
        return ids

    for c in cfg["osm"]["corridors"]:
        o = c["origin_fallback"]; de = c["destination_fallback"]
        dist = _haversine(o[0], o[1], de[0], de[1])
        n = max(int(dist // 200), 6)
        pts = [(o[0] + (de[0] - o[0]) * t / n, o[1] + (de[1] - o[1]) * t / n)
               for t in range(n + 1)]
        corridor_node_ids[c["id"]] = _add_line(pts)

    # connect the two corridors so the graph is one component
    ids_a = corridor_node_ids[cfg["osm"]["corridors"][0]["id"]]
    ids_b = corridor_node_ids[cfg["osm"]["corridors"][1]["id"]]
    best = min(((a, b, _haversine(G.nodes[a]["y"], G.nodes[a]["x"],
                                  G.nodes[b]["y"], G.nodes[b]["x"]))
                for a in ids_a for b in ids_b), key=lambda t: t[2])
    a, b, d = best
    for u, v in ((a, b), (b, a)):
        G.add_edge(u, v, osmid=f"synlink{u}_{v}", length=round(d, 2),
                   highway="tertiary", oneway=False, maxspeed=40, speed_kph=40,
                   name="synthetic connector")
    if logger:
        logger.warning(f"  built OFFLINE synthetic Puducherry graph "
                       f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges) "
                       f"- Overpass unreachable")
    return G


# --------------------------------------------------------------------------- #
def download_puducherry_graph(cfg, logger=None):
    ox = _osmnx()
    o = cfg["osm"]
    out = Path(cfg["_meta"]["repo_root"]) / o["graphml"]
    out.parent.mkdir(parents=True, exist_ok=True)
    center = tuple(o["fallback_center"])
    radius = int(o.get("fallback_radius_m", 5000))
    net = o["network_type"]

    G = None
    method = None
    for url in OVERPASS_MIRRORS:
        _set_overpass(ox, url)
        try:
            G = ox.graph_from_point(center, dist=radius, network_type=net)
            method = f"graph_from_point({center}, dist={radius}) via {url}"
            break
        except Exception as exc:
            _PROV["fallbacks"].append({"stage": "graph_from_point", "mirror": url,
                                       "error": str(exc)[:160]})
            if logger:
                logger.warning(f"  overpass {url} failed ({str(exc)[:80]})")
            continue

    if G is None:
        try:
            _set_overpass(ox, OVERPASS_MIRRORS[0])
            G = ox.graph_from_place(o["place"], network_type=net)
            method = f"graph_from_place({o['place']!r})"
        except Exception as exc:
            _PROV["fallbacks"].append({"stage": "graph_from_place",
                                       "error": str(exc)[:160]})

    synthetic = False
    if G is None:
        G = build_offline_graph(cfg, logger=logger)
        method = "offline_synthetic_fallback"
        synthetic = True

    if not synthetic:
        try:
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
        except Exception:
            pass

    ox.save_graphml(G, out)
    _PROV.update(retrieved_utc=datetime.now(timezone.utc).isoformat(),
                 method=method, synthetic=synthetic,
                 n_nodes=G.number_of_nodes(), n_edges=G.number_of_edges())
    (out.parent / "osm_provenance.json").write_text(json.dumps(_PROV, indent=2))
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


def graph_is_synthetic(cfg) -> bool:
    p = Path(cfg["_meta"]["repo_root"]) / "data" / "osm" / "osm_provenance.json"
    if p.exists():
        try:
            return bool(json.loads(p.read_text()).get("synthetic"))
        except Exception:
            pass
    return False


def provenance() -> dict:
    return dict(_PROV)
