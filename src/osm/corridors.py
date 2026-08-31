"""Build the two study corridors from the Puducherry OSM graph (spec section 13)
and convert each to an ML-ready node/edge graph (spec section 14).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .download import geocode_point


@dataclass
class CorridorResult:
    id: str
    name: str
    origin_latlon: tuple
    destination_latlon: tuple
    route_nodes: list
    route_coords: list          # [[lon, lat], ...] for GeoJSON
    length_m: float
    n_intersections: int
    n_segments: int
    used_fallback_geocode: bool
    ml_graph: dict = field(default_factory=dict)


def _ox():
    import osmnx as ox
    return ox


def _nearest_node(G, lat, lon):
    ox = _ox()
    try:
        return ox.distance.nearest_nodes(G, X=lon, Y=lat)
    except Exception:
        return ox.nearest_nodes(G, lon, lat)


def _shortest_path(G, a, b):
    ox = _ox()
    for fn in ("routing.shortest_path", "shortest_path"):
        obj = ox
        try:
            for part in fn.split("."):
                obj = getattr(obj, part)
            return obj(G, a, b, weight="length")
        except Exception:
            continue
    import networkx as nx
    return nx.shortest_path(G, a, b, weight="length")


def _edge_attr(G, u, v, key="length"):
    data = G.get_edge_data(u, v)
    if not data:
        return None
    d0 = data[min(data.keys())] if isinstance(data, dict) and 0 not in data else data.get(0, data)
    if isinstance(d0, dict):
        return d0.get(key)
    return None


def corridor_to_ml_graph(G, route_nodes: list) -> dict:
    """Node/edge graph with attributes; unavailable OSM tags recorded as null."""
    import networkx as nx
    nodes = []
    for nid in route_nodes:
        nd = G.nodes[nid]
        nodes.append({
            "node_id": int(nid),
            "latitude": nd.get("y"),
            "longitude": nd.get("x"),
            "degree": int(G.degree(nid)),
            "street_count": nd.get("street_count"),
            "intersection_indicator": int(G.degree(nid) >= 3),
        })
    edges = []
    for i, (u, v) in enumerate(zip(route_nodes[:-1], route_nodes[1:])):
        data = G.get_edge_data(u, v) or {}
        d0 = data.get(0, next(iter(data.values()), {})) if data else {}
        maxspeed = d0.get("maxspeed")
        if isinstance(maxspeed, list):
            maxspeed = maxspeed[0]
        try:
            maxspeed_kph = float(str(maxspeed).split()[0]) if maxspeed else None
        except Exception:
            maxspeed_kph = None
        edges.append({
            "edge_id": f"e{i}",
            "start_node": int(u),
            "end_node": int(v),
            "length": d0.get("length"),
            "road_class": d0.get("highway"),
            "speed_limit_kph": maxspeed_kph,
            "speed_est_kph": d0.get("speed_kph"),
            "travel_time_s": d0.get("travel_time"),
            "lanes": d0.get("lanes"),
            "name": d0.get("name"),
        })
    # weighted adjacency (inverse-length) for the ML model
    n = len(nodes)
    A = np.zeros((n, n), np.float32)
    for i, e in enumerate(edges):
        L = e["length"] or 1.0
        A[i, i + 1] = A[i + 1, i] = float(1.0 / max(L, 1.0))
    return {"nodes": nodes, "edges": edges, "adjacency": A.tolist(),
            "n_nodes": n, "n_edges": len(edges)}


def build_corridors(G, cfg, logger=None) -> list[CorridorResult]:
    results: list[CorridorResult] = []
    features = []
    offline = bool(getattr(G, "graph", {}).get("synthetic"))
    if offline and logger:
        logger.warning("  graph is the offline synthetic fallback -> skipping "
                       "Nominatim geocoding, using config endpoint coordinates")
    for c in cfg["osm"]["corridors"]:
        o_lat, o_lon, o_fb = geocode_point(c["origin"], c["origin_fallback"],
                                           logger, offline=offline)
        d_lat, d_lon, d_fb = geocode_point(c["destination"], c["destination_fallback"],
                                           logger, offline=offline)
        na = _nearest_node(G, o_lat, o_lon)
        nb = _nearest_node(G, d_lat, d_lon)
        route = _shortest_path(G, na, nb)
        coords = [[G.nodes[n]["x"], G.nodes[n]["y"]] for n in route]
        # cumulative length
        length = 0.0
        for u, v in zip(route[:-1], route[1:]):
            data = G.get_edge_data(u, v) or {}
            d0 = data.get(0, next(iter(data.values()), {})) if data else {}
            length += float(d0.get("length") or 0.0)
        n_inter = sum(1 for n in route if G.degree(n) >= 3)
        ml = corridor_to_ml_graph(G, route)
        res = CorridorResult(
            id=c["id"], name=c["name"],
            origin_latlon=(o_lat, o_lon), destination_latlon=(d_lat, d_lon),
            route_nodes=[int(n) for n in route], route_coords=coords,
            length_m=round(length, 1), n_intersections=n_inter,
            n_segments=len(route) - 1,
            used_fallback_geocode=bool(o_fb or d_fb), ml_graph=ml,
        )
        results.append(res)
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"id": c["id"], "name": c["name"],
                           "length_m": res.length_m,
                           "n_intersections": n_inter,
                           "n_segments": res.n_segments,
                           "used_fallback_geocode": res.used_fallback_geocode},
        })
        features.append({
            "type": "Feature",
            "geometry": {"type": "MultiPoint",
                         "coordinates": [[o_lon, o_lat], [d_lon, d_lat]]},
            "properties": {"id": c["id"] + "_endpoints",
                           "origin": c["origin"], "destination": c["destination"]},
        })
        if logger:
            logger.info(f"  {c['id']}: {res.length_m:.0f} m, {len(route)} nodes, "
                        f"{n_inter} intersections")

    root = Path(cfg["_meta"]["repo_root"])
    gj = root / cfg["osm"]["corridor_geojson"]
    gj.parent.mkdir(parents=True, exist_ok=True)
    with open(gj, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": features}, fh, indent=2)
    for res in results:
        p = gj.parent / f"{res.id}_ml_graph.json"
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"id": res.id, "name": res.name,
                       "origin_latlon": res.origin_latlon,
                       "destination_latlon": res.destination_latlon,
                       "length_m": res.length_m,
                       "route_nodes": res.route_nodes,
                       "route_coords": res.route_coords,
                       **res.ml_graph}, fh, indent=2)
    return results
