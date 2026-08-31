"""Retrieve the *real* Puducherry driving network from OpenStreetMap.

Auto-download strategy, most reliable first (all bounded by a short timeout):

  1. Direct Overpass HTTP query for a **tight bounding box around the two study
     corridors** (small area -> fast, no Overpass area-limit splitting). The raw
     ``data/osm/puducherry.osm`` it produces also feeds SUMO / ns-3.
  2. ``osmnx.graph_from_bbox`` on the same box, across several Overpass mirrors.
  3. ``graph_from_point`` / ``graph_from_place`` (wider, slower).
  4. Offline anchored fallback graph built from the corridor endpoint coordinates
     in ``config.yaml`` -- ``method: offline_synthetic_fallback`` is recorded in
     ``data/osm/osm_provenance.json`` and every downstream stage still runs.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

_PROV: dict = {"retrieved_utc": None, "method": None, "fallbacks": []}

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]
_DRIVE_HW = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|"
             "living_street|service|road|motorway_link|trunk_link|primary_link|"
             "secondary_link|tertiary_link")


def _osmnx(timeout: int = 30):
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
    base = url.replace("/interpreter", "")
    for attr, val in (("overpass_url", url), ("overpass_endpoint", base)):
        try:
            setattr(ox.settings, attr, val)
        except Exception:
            pass


def _haversine(lat1, lon1, lat2, lon2):
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.asin(math.sqrt(a))


def bbox_from_corridors(cfg, margin_m: float = 1200.0):
    """(south, west, north, east) enclosing every corridor endpoint + margin."""
    lats, lons = [], []
    for c in cfg["osm"]["corridors"]:
        for key in ("origin_fallback", "destination_fallback"):
            lats.append(c[key][0]); lons.append(c[key][1])
    s, n = min(lats), max(lats)
    w, e = min(lons), max(lons)
    dlat = margin_m / 111_320.0
    dlon = margin_m / (111_320.0 * math.cos(math.radians((s + n) / 2)))
    return (s - dlat, w - dlon, n + dlat, e + dlon)


def raw_osm_path(cfg) -> Path:
    return Path(cfg["_meta"]["repo_root"]) / "data" / "osm" / "puducherry.osm"


def bundled_osm_path(cfg) -> Path:
    """Real Puducherry corridor-area OSM extract shipped with the repo
    (data/osm/puducherry.osm.bz2). Used when live Overpass is unreachable so the
    corridors still follow real streets."""
    return Path(cfg["_meta"]["repo_root"]) / "data" / "osm" / "puducherry.osm.bz2"


def _use_bundled_osm(cfg, out_osm: Path, logger=None) -> bool:
    bz = bundled_osm_path(cfg)
    if not bz.exists():
        return False
    try:
        import bz2
        out_osm.parent.mkdir(parents=True, exist_ok=True)
        out_osm.write_bytes(bz2.decompress(bz.read_bytes()))
        if logger:
            logger.warning(f"  live Overpass unreachable -> using bundled real OSM "
                           f"extract {bz.name} ({out_osm.stat().st_size:,} B)")
        return True
    except Exception as exc:  # pragma: no cover
        if logger:
            logger.warning(f"  bundled OSM decompress failed: {exc}")
        return False


# --------------------------------------------------------------------------- #
#  1. direct Overpass HTTP  ->  raw .osm
# --------------------------------------------------------------------------- #
def _direct_overpass(bbox, out_osm: Path, timeout: int, logger=None) -> bool:
    s, w, n, e = bbox
    ql = (f"[out:xml][timeout:{timeout}];"
          f'(way["highway"~"{_DRIVE_HW}"]({s},{w},{n},{e}););'
          f"(._;>;);out body;")
    for url in OVERPASS_MIRRORS:
        try:
            if logger:
                logger.info(f"  overpass POST {url}")
            req = Request(url, data=("data=" + ql).encode(),
                          headers={"User-Agent": "puducherry-vanet-transfer/0.1",
                                   "Content-Type": "application/x-www-form-urlencoded"})
            with urlopen(req, timeout=timeout + 15) as r:
                blob = r.read()
            if len(blob) < 2000 or b"<osm" not in blob[:2000]:
                raise ValueError(f"short/invalid response ({len(blob)} B)")
            out_osm.parent.mkdir(parents=True, exist_ok=True)
            out_osm.write_bytes(blob)
            if logger:
                logger.info(f"      -> {out_osm.name}  {len(blob):,} B")
            return True
        except Exception as exc:  # pragma: no cover - network dependent
            _PROV["fallbacks"].append({"stage": "direct_overpass", "mirror": url,
                                       "error": str(exc)[:160]})
            if logger:
                logger.warning(f"      failed: {str(exc)[:90]}")
    return False


def geocode_point(query: str, fallback_latlon, logger=None, offline: bool = False):
    if offline:
        return float(fallback_latlon[0]), float(fallback_latlon[1]), True
    try:
        lat, lon = _osmnx(timeout=15).geocode(query)
        if logger:
            logger.info(f"  geocoded {query!r} -> ({lat:.5f}, {lon:.5f})")
        return float(lat), float(lon), False
    except Exception as exc:  # pragma: no cover
        if logger:
            logger.warning(f"  geocode failed for {query!r} ({exc}); fallback")
        _PROV["fallbacks"].append({"query": query, "error": str(exc)[:200]})
        return float(fallback_latlon[0]), float(fallback_latlon[1]), True


# --------------------------------------------------------------------------- #
#  offline anchored fallback
# --------------------------------------------------------------------------- #
def build_offline_graph(cfg, logger=None):
    import networkx as nx
    G = nx.MultiDiGraph()
    G.graph.update(crs="epsg:4326", synthetic=True,
                   created_utc=datetime.now(timezone.utc).isoformat())
    nid = [0]
    cids = {}

    def _line(pts, hwy="secondary", spd=40):
        ids = []
        for lat, lon in pts:
            G.add_node(nid[0], x=float(lon), y=float(lat), street_count=2)
            ids.append(nid[0]); nid[0] += 1
        for a, b in zip(ids, ids[1:]):
            d = _haversine(G.nodes[a]["y"], G.nodes[a]["x"], G.nodes[b]["y"], G.nodes[b]["x"])
            for u, v in ((a, b), (b, a)):
                G.add_edge(u, v, osmid=f"syn{u}_{v}", length=round(d, 2), highway=hwy,
                           oneway=False, maxspeed=spd, speed_kph=spd, name="synthetic corridor")
        return ids

    for c in cfg["osm"]["corridors"]:
        o, de = c["origin_fallback"], c["destination_fallback"]
        n = max(int(_haversine(o[0], o[1], de[0], de[1]) // 200), 6)
        cids[c["id"]] = _line([(o[0] + (de[0] - o[0]) * t / n, o[1] + (de[1] - o[1]) * t / n)
                               for t in range(n + 1)])
    a_ids, b_ids = cids[cfg["osm"]["corridors"][0]["id"]], cids[cfg["osm"]["corridors"][1]["id"]]
    a, b, d = min(((a, b, _haversine(G.nodes[a]["y"], G.nodes[a]["x"], G.nodes[b]["y"], G.nodes[b]["x"]))
                   for a in a_ids for b in b_ids), key=lambda t: t[2])
    for u, v in ((a, b), (b, a)):
        G.add_edge(u, v, osmid=f"synlink{u}_{v}", length=round(d, 2), highway="tertiary",
                   oneway=False, maxspeed=40, speed_kph=40, name="synthetic connector")
    if logger:
        logger.warning(f"  built OFFLINE synthetic graph ({G.number_of_nodes()} nodes, "
                       f"{G.number_of_edges()} edges) - Overpass unreachable")
    return G


# --------------------------------------------------------------------------- #
def _graph_from_bbox(ox, bbox, net):
    s, w, n, e = bbox
    try:                                   # osmnx >= 2.0
        return ox.graph_from_bbox(bbox=(w, s, e, n), network_type=net)
    except TypeError:
        return ox.graph_from_bbox(n, s, e, w, network_type=net)


def download_puducherry_graph(cfg, logger=None):
    ox = _osmnx()
    o = cfg["osm"]
    out = Path(cfg["_meta"]["repo_root"]) / o["graphml"]
    out.parent.mkdir(parents=True, exist_ok=True)
    net = o["network_type"]
    bbox = bbox_from_corridors(cfg, float(o.get("bbox_margin_m", 1200)))
    raw = raw_osm_path(cfg)

    G, method = None, None

    # 1. direct Overpass -> raw .osm -> osmnx graph_from_xml
    if _direct_overpass(bbox, raw, timeout=45, logger=logger):
        try:
            G = ox.graph_from_xml(raw, retain_all=False)
            method = f"direct_overpass_bbox {tuple(round(x, 4) for x in bbox)}"
        except Exception as exc:
            _PROV["fallbacks"].append({"stage": "graph_from_xml", "error": str(exc)[:160]})

    # 2. osmnx graph_from_bbox across mirrors
    if G is None:
        for url in OVERPASS_MIRRORS:
            _set_overpass(ox, url)
            try:
                G = _graph_from_bbox(ox, bbox, net)
                method = f"graph_from_bbox via {url}"
                break
            except Exception as exc:
                _PROV["fallbacks"].append({"stage": "graph_from_bbox", "mirror": url,
                                           "error": str(exc)[:140]})

    # 3. point / place
    if G is None:
        center = tuple(o["fallback_center"])
        for url in OVERPASS_MIRRORS[:3]:
            _set_overpass(ox, url)
            try:
                G = ox.graph_from_point(center, dist=int(o.get("fallback_radius_m", 5000)),
                                        network_type=net)
                method = f"graph_from_point via {url}"
                break
            except Exception as exc:
                _PROV["fallbacks"].append({"stage": "graph_from_point", "error": str(exc)[:140]})
    if G is None:
        try:
            _set_overpass(ox, OVERPASS_MIRRORS[0])
            G = ox.graph_from_place(o["place"], network_type=net)
            method = f"graph_from_place({o['place']!r})"
        except Exception as exc:
            _PROV["fallbacks"].append({"stage": "graph_from_place", "error": str(exc)[:140]})

    # 4. bundled real OSM extract (offline but still real streets)
    if G is None and _use_bundled_osm(cfg, raw, logger=logger):
        try:
            G = ox.graph_from_xml(raw, retain_all=False)
            method = "bundled_osm_extract (data/osm/puducherry.osm.bz2)"
        except Exception as exc:
            _PROV["fallbacks"].append({"stage": "bundled_graph_from_xml",
                                       "error": str(exc)[:160]})

    synthetic = G is None
    if synthetic:
        G = build_offline_graph(cfg, logger=logger)
        method = "offline_synthetic_fallback"
    else:
        try:
            G = ox.add_edge_speeds(G); G = ox.add_edge_travel_times(G)
        except Exception:
            pass

    try:
        ox.save_graphml(G, out)
    except Exception as exc:
        if logger:
            logger.warning(f"  save_graphml failed ({str(exc)[:70]}); pickle is authoritative")
    import pickle as _pk
    with open(out.with_suffix(".pkl"), "wb") as fh:
        _pk.dump(G, fh)

    _PROV.update(retrieved_utc=datetime.now(timezone.utc).isoformat(), method=method,
                 synthetic=synthetic, bbox=[round(x, 5) for x in bbox],
                 raw_osm=str(raw) if raw.exists() else None,
                 n_nodes=G.number_of_nodes(), n_edges=G.number_of_edges())
    (out.parent / "osm_provenance.json").write_text(json.dumps(_PROV, indent=2))
    if logger:
        logger.info(f"  saved {out.name}  ({G.number_of_nodes()} nodes, "
                    f"{G.number_of_edges()} edges)  via {method}")
    return G


def load_graph(cfg):
    path = Path(cfg["_meta"]["repo_root"]) / cfg["osm"]["graphml"]
    pkl = path.with_suffix(".pkl")
    if pkl.exists():
        import pickle as _pk
        with open(pkl, "rb") as fh:
            return _pk.load(fh)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/download_osm.py")
    return _osmnx().load_graphml(path)


def graph_is_synthetic(cfg) -> bool:
    p = Path(cfg["_meta"]["repo_root"]) / "data" / "osm" / "osm_provenance.json"
    try:
        return bool(json.loads(p.read_text()).get("synthetic")) if p.exists() else False
    except Exception:
        return False


def provenance() -> dict:
    return dict(_PROV)
