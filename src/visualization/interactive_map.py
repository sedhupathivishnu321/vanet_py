"""Interactive Folium map of the Puducherry VANET scenario (spec section 50)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def build_interactive_map(cfg, vanet_npz: str | None = None,
                          out_html: str | None = None, logger=None) -> str:
    import folium

    root = Path(cfg["_meta"]["repo_root"])
    out_html = out_html or str(root / cfg["project"]["outputs_dir"]
                               / "maps" / "puducherry_vanet_map.html")
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)

    center = cfg["osm"]["fallback_center"]
    gj_path = root / cfg["osm"]["corridor_geojson"]
    m = folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")

    # data-provenance banner (spec section 51)
    banner = ('<div style="position:fixed;top:8px;left:8px;z-index:9999;'
              'background:white;padding:8px 12px;border:1px solid #888;'
              'font:12px sans-serif;border-radius:6px">'
              '<b>SOURCE DATASET:</b> REAL PUBLIC DATA (METR-LA/PEMS-BAY)<br>'
              '<b>ROAD NETWORK:</b> OpenStreetMap (real geometry)<br>'
              '<b>PUDUCHERRY TARGET TRAFFIC:</b> SIMULATED<br>'
              '<b>VANET COMMUNICATION:</b> SIMULATED (PDR / latency / AoI)</div>')
    m.get_root().html.add_child(folium.Element(banner))

    colors = {"corridor_1": "#d1495b", "corridor_2": "#2a9d8f"}
    if gj_path.exists():
        gj = json.loads(gj_path.read_text())
        for feat in gj["features"]:
            props = feat["properties"]
            if feat["geometry"]["type"] == "LineString":
                latlon = [[c[1], c[0]] for c in feat["geometry"]["coordinates"]]
                folium.PolyLine(
                    latlon, color=colors.get(props["id"], "#333"), weight=5,
                    opacity=0.8,
                    tooltip=f"{props['name']} ({props.get('length_m', '?')} m)"
                ).add_to(m)
                folium.Marker(latlon[0], tooltip=f"{props['name']} - origin",
                              icon=folium.Icon(color="green")).add_to(m)
                folium.Marker(latlon[-1], tooltip=f"{props['name']} - destination",
                              icon=folium.Icon(color="red")).add_to(m)

    # place labels
    for c in cfg["osm"]["corridors"]:
        for role, key in [("origin", "origin_fallback"),
                          ("destination", "destination_fallback")]:
            ll = c[key]
            folium.CircleMarker(ll, radius=4, color="#555", fill=True,
                                tooltip=f"{c[role]}").add_to(m)

    # vehicle layer from the default VANET run
    if vanet_npz and Path(vanet_npz).exists():
        d = np.load(vanet_npz, allow_pickle=True)
        try:
            coords = d["corridor_coords"]           # (P, 2) lon,lat
            frame = d["last_frame"]                  # (n,7)
            connected = d["last_frame_connected"]    # (n,)
            L = float(d["corridor_length_m"])
            fg = folium.FeatureGroup(name="vehicles (last snapshot)")
            for row, conn in zip(frame, connected):
                frac = np.clip(row[2] / max(L, 1.0), 0, 1)
                i = int(frac * (len(coords) - 1))
                lat, lon = coords[i][1], coords[i][0]
                folium.CircleMarker(
                    [lat, lon], radius=3,
                    color="#1f77b4" if conn else "#bbbbbb",
                    fill=True, fill_opacity=0.9,
                    tooltip=f"veh {int(row[0])} v={row[3]:.1f} m/s "
                            f"{'connected' if conn else 'not connected'}"
                ).add_to(fg)
            fg.add_to(m)
        except Exception as exc:  # pragma: no cover
            if logger:
                logger.warning(f"vehicle layer skipped: {exc}")

    folium.LayerControl().add_to(m)
    m.save(out_html)
    if logger:
        logger.info(f"  interactive map -> {out_html}")
    return out_html
