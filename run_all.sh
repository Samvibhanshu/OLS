#!/usr/bin/env bash
# Reproduces every number and figure in results/.   Runtime about 10 min on 2 CPU cores.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m pytest -q tests
cd experiments
python3 exp1_contamination.py   # E1 sweep, E2 ablation, E3 lineage quality
python3 exp2_rollback.py        # E4 50k-record selective rollback vs full reprocessing
python3 exp3_scaling.py         # E5 cost scaling
python3 exp4_retroactive.py     # E6 retroactive correction of published daily index
python3 exp5_sensitivity.py     # E7 threshold / exponent sensitivity + validation tuning
python3 exp6_rag_quality.py     # R1 answer quality vs contamination, R2 weight ablation, R3 grounding audit
python3 exp7_rag_correction.py  # R4 retroactive correction of the RAG system (sleeper feed exposed)
python3 exp8_rag_stress.py      # R5 text-disguise stress test
python3 exp9_rag_latency.py     # R6 latency / retraction cost
cd ..
python3 examples_demo.py
