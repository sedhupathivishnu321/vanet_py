#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Codespace / devcontainer bootstrap.
# Installs the SUMO binaries (apt) and the Python dependencies (pip).
# Safe to re-run.
# ---------------------------------------------------------------------------
set -e

echo "=============================================================="
echo " puducherry-vanet-transfer : environment bootstrap"
echo "=============================================================="

# --- 1. System packages: SUMO + geospatial libs -----------------------------
echo "[setup] installing system packages (sudo apt) ..."
sudo apt-get update -y
# 'sumo', 'sumo-tools' live in Debian/Ubuntu 'universe'/main. GDAL/proj are
# pulled for geopandas wheels that need system libs on some images.
sudo apt-get install -y --no-install-recommends \
    sumo sumo-tools sumo-doc \
    libgdal-dev gdal-bin \
    libspatialindex-dev \
    || echo "[setup] WARNING: apt install of SUMO failed - project will fall back to the built-in micro-simulator."

# --- 2. SUMO_HOME ----------------------------------------------------------
if [ -d /usr/share/sumo ]; then
  export SUMO_HOME=/usr/share/sumo
  echo "export SUMO_HOME=/usr/share/sumo" >> "$HOME/.bashrc"
  echo "[setup] SUMO_HOME=$SUMO_HOME"
  sumo --version 2>/dev/null | head -n 1 || true
else
  echo "[setup] SUMO not found under /usr/share/sumo - continuing without it."
fi

# --- 3. Python dependencies ---------------------------------------------------
echo "[setup] upgrading pip ..."
python -m pip install --upgrade pip wheel setuptools

# CPU-only PyTorch wheel first (smaller / faster than the default CUDA wheel;
# Codespaces has no GPU). Remove this line to let requirements.txt pick the
# default wheel, e.g. if you attach a GPU.
echo "[setup] installing CPU PyTorch ..."
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1" \
    || python -m pip install "torch>=2.1"

echo "[setup] installing Python requirements ..."
python -m pip install -r requirements.txt

# --- 4. Make 'src' importable everywhere -----------------------------------
python -m pip install -e . || echo "[setup] editable install skipped (no pyproject/setup) - PYTHONPATH is set instead."

# --- 5. Sanity check --------------------------------------------------------
echo "[setup] verifying environment ..."
python scripts/check_env.py || true

echo "=============================================================="
echo " Bootstrap complete."
echo " Next:  python run_all.py --quick"
echo "=============================================================="
