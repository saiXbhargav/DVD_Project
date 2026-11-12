#!/usr/bin/env bash
# Regenerate every result from the model file, in order, with fixed seeds.
#   ./run_all.sh            all steps
#   ./run_all.sh 3 6 7 8    only the listed steps
# Each step logs to results/log_<step>.txt. Safe to run under nohup:
#   nohup ./run_all.sh > results/run_all.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(1 2 3 4 5 6 7 8)
for s in "${steps[@]}"; do
    script=$(ls 0${s}_*.py)
    echo "=== $(date '+%F %T')  step $s: $script"
    python3 -u "$script" > "results/log_${s}.txt" 2>&1
    rc=$?
    tail -n 30 "results/log_${s}.txt"
    echo "=== $(date '+%F %T')  step $s finished (exit $rc)"
done
echo "=== ALL DONE"
