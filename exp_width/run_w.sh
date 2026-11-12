#!/bin/bash
# Varying-width evaluation: SPICE references, then ML versions A and B. Resumable.
# Log: exp_width/run_w.log
cd "$(dirname "$0")" || exit 1
set -e
echo "===== spice_w $(date)"
python3 -u spice_w.py
echo "===== eval_w $(date)"
python3 -u eval_w.py
echo "===== finished $(date)"
