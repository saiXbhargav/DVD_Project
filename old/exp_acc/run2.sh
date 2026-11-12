#!/bin/bash
# exp_acc round 2 (resumable); log in exp_acc/run2.log, per-case MLP logs in exp_acc/mlp_<case>.log
cd "$(dirname "$0")" || exit 1
set -e
echo "===== step1_gpx $(date)"
python3 -u step1_gpx.py
echo "===== step2_mlp (4 cases in parallel, 1 thread each) $(date)"
for c in nmos_on nmos_off pmos_on pmos_off; do
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -u step2_mlp.py "$c" > "mlp_$c.log" 2>&1 &
done
wait
grep -h "MLP87\|exists\|Error" mlp_*.log || true
echo "===== evaluate2 $(date)"
python3 -u evaluate2.py
echo "===== finished $(date)"
