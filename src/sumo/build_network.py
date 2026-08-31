"""OSM -> SUMO network conversion (spec section 17).

Wraps ``netconvert``.  If SUMO is not installed the project falls back to the
built-in micro-simulator; this module just reports that cleanly.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def sumo_available() -> bool:
    return shutil.which("sumo") is not None and shutil.which("netconvert") is not None


def sumo_version() -> str | None:
    if not sumo_available():
        return None
    try:
        out = subprocess.check_output(["sumo", "--version"], text=True,
                                      stderr=subprocess.DEVNULL)
        return out.splitlines()[0].strip()
    except Exception:
        return None


def build_sumo_network(cfg, logger=None) -> dict:
    """Try to convert data/osm/puducherry.graphml's source OSM extract into a
    SUMO .net.xml.  Requires an .osm file; if only a .graphml is present we
    export it via osmnx first.  Returns a status dict."""
    root = Path(cfg["_meta"]["repo_root"])
    osm_dir = root / "data" / "osm"
    net_xml = root / cfg["sumo"]["net_xml"]
    net_xml.parent.mkdir(parents=True, exist_ok=True)
    status = {"sumo_available": sumo_available(), "sumo_version": sumo_version(),
              "net_xml": str(net_xml), "built": False, "note": ""}

    if not sumo_available():
        status["note"] = ("SUMO binaries not found - target simulation will use "
                           "the built-in IDM micro-simulator.")
        if logger:
            logger.warning(status["note"])
        return status

    osm_raw = osm_dir / "puducherry.osm"
    if not osm_raw.exists():
        # Export a bounding-box .osm via OSMnx graph -> not directly supported;
        # instead ask netconvert to consume the graphml-derived polygon is not
        # possible. We download a raw OSM extract with osmnx utilities.
        try:
            import osmnx as ox
            o = cfg["osm"]
            gdf = None
            try:
                poly = ox.geocode_to_gdf(o["place"]).geometry.iloc[0]
                bbox = poly.bounds  # (minx, miny, maxx, maxy)
            except Exception:
                c = o["fallback_center"]
                r = o["fallback_radius_m"] / 111_000.0
                bbox = (c[1] - r, c[0] - r, c[1] + r, c[0] + r)
            # use SUMO's osmGet if available
            osm_get = None
            sumo_home = Path(__import__("os").environ.get("SUMO_HOME", "/usr/share/sumo"))
            cand = sumo_home / "tools" / "osmGet.py"
            if cand.exists():
                osm_get = str(cand)
            if osm_get:
                subprocess.run(["python", osm_get,
                                f"--bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                                "--output-dir", str(osm_dir),
                                "--prefix", "puducherry"], check=True)
                cands = list(osm_dir.glob("puducherry*.osm*"))
                if cands:
                    osm_raw = cands[0]
        except Exception as exc:
            status["note"] = f"could not fetch raw .osm ({exc}); using IDM fallback"
            if logger:
                logger.warning(status["note"])
            return status

    if osm_raw.exists():
        try:
            subprocess.run([
                "netconvert", "--osm-files", str(osm_raw), "-o", str(net_xml),
                "--geometry.remove", "--roundabouts.guess",
                "--ramps.guess", "--junctions.join",
                "--tls.guess-signals", "--tls.discard-simple",
                "--tls.join", "--no-turnarounds", "true",
            ], check=True)
            status["built"] = net_xml.exists()
            status["note"] = "netconvert succeeded"
            if logger and status["built"]:
                logger.info(f"  SUMO network -> {net_xml}")
        except Exception as exc:
            status["note"] = f"netconvert failed ({exc}); using IDM fallback"
            if logger:
                logger.warning(status["note"])
    return status
