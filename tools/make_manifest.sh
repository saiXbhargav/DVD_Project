#!/usr/bin/env bash
# Write MANIFEST.sha256: SHA-256 of every input artifact of the final results (data, models,
# SPICE references) and of the reference outputs. verify_final.py checks against it.
set -euo pipefail
cd "$(dirname "$0")/.."
{
    ls data/nmos_on.csv data/nmos_off.csv data/pmos_on.csv data/pmos_off.csv data/corners.csv data/corners_w.csv
    ls data/train_extra/*.csv
    ls models_trained/*.joblib
    ls results/gate_spice_w.csv results/circuit_spice_w.csv results/gate_spice.csv results/circuit_spice.csv
    ls expected_results/*.csv
} | xargs sha256sum > MANIFEST.sha256
wc -l MANIFEST.sha256
