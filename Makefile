# Convenience wrapper around the pipeline scripts.
# `make quick` runs the whole thing at reduced scale; `make full` runs the
# complete experiment matrix.

PY ?= python
PROFILE ?= --quick

.PHONY: help env quick full test clean dashboard ns3-setup \
        dataset inspect osm routes sumo train transfer vanet \
        eval-pred control eval-control report figures map

help:
	@echo "targets:"
	@echo "  make env           - print environment / dependency report"
	@echo "  make quick         - full pipeline, reduced scale (~15-25 min CPU)"
	@echo "  make full          - full pipeline, complete experiment matrix"
	@echo "  make test          - run pytest"
	@echo "  make dashboard     - launch the Streamlit dashboard on :8501"
	@echo "  make clean         - remove regenerated data/model/output artefacts"
	@echo "  individual stages: dataset inspect osm routes sumo train transfer"
	@echo "                     vanet eval-pred control eval-control report"

env:
	$(PY) scripts/check_env.py

quick:
	$(PY) run_all.py --quick

full:
	$(PY) run_all.py --full

test:
	$(PY) -m pytest -q

dashboard:
	streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0

ns3-setup:
	bash ns3/setup_ns3.sh
	@echo "then set  vanet.backend: ns3  in config.yaml"

dataset:
	$(PY) scripts/download_dataset.py $(PROFILE)
inspect:
	$(PY) scripts/inspect_dataset.py $(PROFILE)
osm:
	$(PY) scripts/download_osm.py $(PROFILE)
routes:
	$(PY) scripts/build_puducherry_routes.py $(PROFILE)
sumo:
	$(PY) scripts/build_sumo.py $(PROFILE)
train:
	$(PY) scripts/train_source.py $(PROFILE)
transfer:
	$(PY) scripts/transfer_target.py $(PROFILE)
vanet:
	$(PY) scripts/simulate_vanet.py $(PROFILE)
eval-pred:
	$(PY) scripts/evaluate_prediction.py $(PROFILE)
control:
	$(PY) scripts/train_control.py $(PROFILE)
eval-control:
	$(PY) scripts/evaluate_control.py $(PROFILE)
report:
	$(PY) scripts/generate_report.py $(PROFILE)

clean:
	rm -rf data/source/* data/osm/* data/sumo/* data/target/* data/vanet/* data/processed/*
	rm -rf models/checkpoints/* models/baselines/*
	rm -rf outputs/figures/* outputs/tables/* outputs/maps/* outputs/logs/* outputs/reports/*
	find . -name .gitkeep -path '*/data/*' -exec touch {} +
	@echo "cleaned."
