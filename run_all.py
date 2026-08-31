#!/usr/bin/env python
"""End-to-end pipeline runner (spec sections 56, 64-65).

    python run_all.py --quick      # reduced scale, ~15-25 min on a 4-core CPU
    python run_all.py --full       # complete experiment matrix
    python run_all.py --smoke      # CI wiring check (tiny)
    python run_all.py --from train_source --quick   # resume from a stage
    python run_all.py --only vanet,eval-control --quick

Every stage is a standalone script under scripts/; this runner just sequences
them, forwards the profile flag, logs to outputs/logs/run_all_<ts>.log and
prints a timing + pass/fail summary.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent

STAGES = [
    ("check_env",        "scripts/check_env.py",               False),
    ("download_dataset", "scripts/download_dataset.py",        True),
    ("inspect_dataset",  "scripts/inspect_dataset.py",         True),
    ("download_osm",     "scripts/download_osm.py",            True),
    ("build_routes",     "scripts/build_puducherry_routes.py", True),
    ("build_sumo",       "scripts/build_sumo.py",              True),
    ("train_source",     "scripts/train_source.py",            True),
    ("transfer_target",  "scripts/transfer_target.py",         True),
    ("simulate_vanet",   "scripts/simulate_vanet.py",          True),
    ("evaluate_prediction", "scripts/evaluate_prediction.py",  True),
    ("train_control",    "scripts/train_control.py",           True),
    ("evaluate_control", "scripts/evaluate_control.py",        True),
    ("tests",            "-m pytest -q",                       False),
    ("generate_report",  "scripts/generate_report.py",         True),
]
ALIAS = {"vanet": "simulate_vanet", "eval-pred": "evaluate_prediction",
         "control": "train_control", "eval-control": "evaluate_control",
         "report": "generate_report", "routes": "build_routes",
         "osm": "download_osm", "dataset": "download_dataset",
         "inspect": "inspect_dataset", "sumo": "build_sumo",
         "train": "train_source", "transfer": "transfer_target"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    g.add_argument("--smoke", action="store_true")
    ap.add_argument("--from", dest="from_stage", help="start at this stage")
    ap.add_argument("--only", help="comma list of stages to run")
    ap.add_argument("--skip", default="", help="comma list of stages to skip")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    profile_flag = ("--quick" if args.quick else "--smoke" if args.smoke
                    else "--full" if args.full else "--quick")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logdir = REPO / "outputs" / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    runlog = logdir / f"run_all_{ts}.log"

    def norm(name):
        return ALIAS.get(name.strip(), name.strip())

    only = {norm(s) for s in args.only.split(",")} if args.only else None
    skip = {norm(s) for s in args.skip.split(",") if s.strip()}

    selected = []
    started = args.from_stage is None
    for name, script, takes_profile in STAGES:
        if args.from_stage and norm(args.from_stage) == name:
            started = True
        if only is not None:
            if name in only:
                selected.append((name, script, takes_profile))
            continue
        if not started or name in skip:
            continue
        selected.append((name, script, takes_profile))

    print(f"profile          : {profile_flag}")
    print(f"stages to run    : {[s[0] for s in selected]}")
    print(f"run log          : {runlog}\n")
    if args.dry_run:
        return 0

    results = []
    t_all = time.time()
    with open(runlog, "w", encoding="utf-8") as lf:
        for name, script, takes_profile in selected:
            cmd = [sys.executable, *script.split()] if script.startswith("-m") \
                else [sys.executable, str(REPO / script)]
            if takes_profile and profile_flag != "--full":
                cmd.append(profile_flag)
            banner = f"\n{'='*70}\n[{name}] {' '.join(cmd)}\n{'='*70}"
            print(banner); lf.write(banner + "\n"); lf.flush()
            t0 = time.time()
            try:
                proc = subprocess.run(cmd, cwd=REPO, text=True,
                                      capture_output=True)
                lf.write(proc.stdout + "\n" + proc.stderr + "\n"); lf.flush()
                # echo a tail of stdout so the console stays informative
                tail = "\n".join(proc.stdout.strip().splitlines()[-25:])
                print(tail)
                ok = proc.returncode == 0
                if not ok:
                    print(f"  !! {name} exited {proc.returncode}")
                    print("\n".join(proc.stderr.strip().splitlines()[-20:]))
            except Exception as exc:  # pragma: no cover
                ok = False
                lf.write(f"EXCEPTION: {exc}\n")
                print(f"  !! {name} raised {exc}")
            dt = time.time() - t0
            results.append((name, ok, round(dt, 1)))
            if not ok and args.stop_on_error:
                break

    print(f"\n{'='*70}\nSUMMARY  (total {round(time.time()-t_all,1)} s)\n{'='*70}")
    for name, ok, dt in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name:22s} {dt:8.1f}s")
    failed = [n for n, ok, _ in results if not ok]
    if failed:
        print(f"\nfailed stages: {failed}  (see {runlog})")
    print("\noutputs/  tables, figures, maps, reports, logs")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
