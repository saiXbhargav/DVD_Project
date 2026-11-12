#!/bin/bash
# Fix 1 (gate-voltage-aware models). Resumable. Log: exp_vg/run_vg.log
#   1. off-rail-gate training / validation data
#   2. evaluation with the CURRENT final models + real gate voltages (no retraining)  -> out/vg_*_gpx.csv
#   3. training of the gate-voltage-aware models + transistor validation
#   4. evaluation with the new models                                               -> out/vg_*.csv
cd "$(dirname "$0")" || exit 1
set -e
echo "===== gen_vg $(date)";                 python3 -u gen_vg.py
echo "===== eval_vg, current models $(date)"; MODELSET=GPX python3 -u eval_vg.py
echo "===== train_vg $(date)";               python3 -u train_vg.py
echo "===== eval_vg, new models $(date)";     MODELSET=VG python3 -u eval_vg.py
echo "===== finished $(date)"
