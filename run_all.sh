#!/usr/bin/env bash
# Final pipeline: regenerate every result in order, then check it against the reported numbers.
#   ./run_all.sh            steps 1-9, then verify_final.py
#   ./run_all.sh 6 8 9      only the listed steps (then verify)
#   ./run_all.sh 10         optional voltage-aware parity checker (~3 h)
#   SMOKE=1 ./run_all.sh 5 6 7 8 9    quick test on 3 corners (results/smoke/)
# Steps 1, 3, 4, 5, 7 keep existing outputs (data, models, model comparison, SPICE references:
# skip-if-exists), so with the shipped artifacts only steps 2, 6, 8, 9 do real work (~2 h).
# Delete those files to regenerate them with the same seeds (hours each).
# Logs: results/logs/step_<n>.log.
#   nohup ./run_all.sh > run_all.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/logs
steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(1 2 3 4 5 6 7 8 9)
for s in "${steps[@]}"; do
    script=$(ls "$(printf '%02d' "$s")"_*.py)
    echo "=== $(date '+%F %T')  step $s: $script"
    python3 -u "$script" 2>&1 | tee "results/logs/step_${s}.log"
    echo "=== $(date '+%F %T')  step $s finished"
done
if [ "${SMOKE:-0}" != "1" ]; then
    python3 -u verify_final.py 2>&1 | tee results/logs/verify.log
fi
echo "=== ALL DONE"
