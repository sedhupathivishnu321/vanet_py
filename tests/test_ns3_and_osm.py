"""ns-3 VANET backend (Python side) + OSM bbox / bundled-extract fallback."""
import numpy as np
import pytest

from src.osm import bbox_from_corridors, bundled_osm_path, raw_osm_path
from src.osm.download import _DRIVE_HW, OVERPASS_MIRRORS
from src.sumo.idm_sim import IDMSimulator
from src.experiments.target_scenario import load_corridor, run_scenario
from src.vanet import (CommunicationChannel, build_partial_observation,
                       assemble_partial_obs, export_ns2_mobility, ns3_available,
                       Beacon, Ns3Unavailable)
from src.vanet.mobility_export import vehicle_state_at
from src.vanet.ns3_channel import ns3_dir, _parse_trace


def test_bbox_encloses_all_corridor_endpoints(cfg):
    s, w, n, e = bbox_from_corridors(cfg, margin_m=0.0)
    for c in cfg["osm"]["corridors"]:
        for key in ("origin_fallback", "destination_fallback"):
            lat, lon = c[key]
            assert s <= lat <= n and w <= lon <= e
    # margin actually widens the box
    s2, w2, n2, e2 = bbox_from_corridors(cfg, margin_m=1000.0)
    assert s2 < s and n2 > n and w2 < w and e2 > e


def test_bundled_osm_extract_present_and_valid(cfg):
    bz = bundled_osm_path(cfg)
    assert bz.exists() and bz.stat().st_size > 50_000
    import bz2
    raw = bz2.decompress(bz.read_bytes())
    assert raw[:200].lstrip().startswith(b"<?xml") or b"<osm" in raw[:400]
    assert raw.count(b"<way ") > 500 and raw.count(b"<node ") > 2000


def test_overpass_query_targets_drivable_roads():
    assert "residential" in _DRIVE_HW and "motorway" in _DRIVE_HW
    assert all(m.endswith("/interpreter") for m in OVERPASS_MIRRORS)


def test_ns3_not_built_falls_back_gracefully(cfg, monkeypatch):
    # force "ns-3 requested but unavailable"
    cfg2 = dict(cfg)
    cfg2["vanet"] = dict(cfg["vanet"]); cfg2["vanet"]["backend"] = "ns3"
    monkeypatch.setenv("NS3_DIR", "/nonexistent/ns3")
    assert ns3_available(cfg2) is False
    sc = run_scenario(cfg2, "corridor_1", seed=1, demand_level="LOW")
    assert sc["comm_backend"] == "analytic"          # fell back, did not crash
    assert sc["partial_obs"].observed_state.shape == sc["partial_obs"].ground_truth_state.shape


def test_mobility_export_ns2_format(cfg):
    sim = IDMSimulator(cfg, load_corridor(cfg, "corridor_1"), "MEDIUM", seed=2).run()
    mob = export_ns2_mobility(sim, cfg["_meta"]["repo_root"] + "/data/vanet/_t.tcl",
                              penetration=0.5, seed=2)
    txt = mob.path.read_text()
    assert mob.n_nodes >= 1
    assert "$node_(0) set X_" in txt and 'setdest' in txt and '$ns_ at ' in txt
    # every referenced node index is < n_nodes
    import re
    idxs = {int(i) for i in re.findall(r"\$node_\((\d+)\)", txt)}
    assert max(idxs) < mob.n_nodes
    assert set(mob.node_to_veh) == idxs


def test_assemble_partial_obs_from_beacons_matches_analytic_shape(cfg):
    sim = IDMSimulator(cfg, load_corridor(cfg, "corridor_1"), "MEDIUM", seed=3).run()
    # analytic
    ch = CommunicationChannel(0.4, 0.9, 50, seed=3)
    po_a = build_partial_observation(sim, ch, cfg)
    # ns-3-style: fabricate delivered beacons from the true tracks
    rng = np.random.default_rng(0)
    beacons = []
    for k, t in enumerate(sim.times):
        for row in sim.vehicle_frames[k]:
            if rng.random() < 0.5:
                st = vehicle_state_at(sim, int(row[0]), float(t))
                if st:
                    beacons.append(Beacon(int(row[0]), st[0], st[2], st[1], 0.0,
                                          sim.heading_deg, float(t),
                                          float(t) + rng.uniform(0.001, 0.03)))
    po_n = assemble_partial_obs(beacons, sim, cfg, penetration=0.5, pdr=0.5,
                                latency_ms=15.0, backend="ns3")
    assert po_n.observed_state.shape == po_a.observed_state.shape
    assert po_n.backend == "ns3" and po_a.backend == "analytic"
    assert 0.0 <= po_n.mask.mean() <= 1.0


def test_parse_trace_reads_csv(tmp_path):
    p = tmp_path / "rx.csv"
    p.write_text("event,t,gen_t,node,peer,dist_m\n"
                 "tx,1.0,1.0,3,-1,0\n"
                 "rx,1.02,1.0,5,3,140.2\n")
    df = _parse_trace(p)
    assert list(df["event"]) == ["tx", "rx"]
    assert float(df[df.event == "rx"]["dist_m"].iloc[0]) == pytest.approx(140.2)
