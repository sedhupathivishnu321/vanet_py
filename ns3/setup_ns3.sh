#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build ns-3 with the modules the VANET beacon scenario needs, and install
# ns3/vanet-beacon.cc into <ns-3>/scratch/.  Idempotent.
#
#   bash ns3/setup_ns3.sh            # clone + build into ./ns-3-dev
#   NS3_DIR=/opt/ns-3.42 bash ns3/setup_ns3.sh   # use an existing checkout
#
# After it finishes:  export NS3_DIR=$PWD/ns-3-dev   (also appended to ~/.bashrc)
# The Python pipeline auto-detects $NS3_DIR; set vanet.backend: ns3 in config.yaml
# (or leave it 'analytic' -- the pipeline falls back automatically if ns-3 is
# missing).
# ---------------------------------------------------------------------------
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
NS3_DIR="${NS3_DIR:-$REPO/ns-3-dev}"
NS3_BRANCH="${NS3_BRANCH:-ns-3.42}"

echo "=============================================================="
echo " ns-3 setup   ->  $NS3_DIR   (branch $NS3_BRANCH)"
echo "=============================================================="

if [ ! -x "$NS3_DIR/ns3" ]; then
  echo "[ns3] cloning ..."
  sudo apt-get update -y || true
  sudo apt-get install -y g++ cmake ninja-build python3 git ccache \
       libgsl-dev libsqlite3-dev pkg-config || true
  git clone --depth 1 --branch "$NS3_BRANCH" \
      https://gitlab.com/nsnam/ns-3-dev.git "$NS3_DIR" \
    || git clone --depth 1 https://github.com/nsnam/ns-3-dev.git "$NS3_DIR"
fi

cp "$HERE/vanet-beacon.cc" "$NS3_DIR/scratch/vanet-beacon.cc"
echo "[ns3] installed scratch/vanet-beacon.cc"

cd "$NS3_DIR"
./ns3 configure -d optimized \
  --enable-modules=core,network,internet,mobility,wifi,wave,applications,propagation,stats \
  --disable-python --disable-examples --disable-tests
./ns3 build vanet-beacon

if ! grep -q "NS3_DIR=$NS3_DIR" "$HOME/.bashrc" 2>/dev/null; then
  echo "export NS3_DIR=$NS3_DIR" >> "$HOME/.bashrc"
fi
export NS3_DIR

echo "=============================================================="
echo " ns-3 ready.  quick check:"
./ns3 run "vanet-beacon --PrintHelp" 2>/dev/null | head -n 5 || true
echo
echo " Next:  set  vanet.backend: ns3  in config.yaml, then"
echo "        python scripts/simulate_vanet.py --quick"
echo "=============================================================="
