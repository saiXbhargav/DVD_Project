#!/bin/bash
# Waits until ml_nodes.py (version B, fixed widths) has saved its OFFICIAL-corner results,
# then stops it (it would continue with the slow sequential fresh-corner part), runs the fresh
# corners with the parallel solver, and then the varying-width evaluation.
# Log: exp_acc/after_run1.log
cd "$(dirname "$0")" || exit 1
START=$(date +%s)
echo "===== waiting for the official version B results $(date)"
while true; do
    f=out/mlnodes_summary.csv
    if [ -f "$f" ] && [ "$(stat -c %Y "$f")" -ge "$START" ]; then break; fi
    if ! pgrep -f "python3 -u ml_nodes.py" > /dev/null; then echo "ml_nodes.py is no longer running"; break; fi
    sleep 60
done
sleep 20
echo "===== official results saved; stopping ml_nodes.py $(date)"
pkill -f "python3 -u ml_nodes.py" || true
sleep 5
grep -E "^official version" mlnodes.log
echo "===== fresh corners, parallel $(date)"
python3 -u ml_nodes_par.py || echo "!!!!! ml_nodes_par FAILED"
echo "===== varying widths $(date)"
(cd ../exp_width && ./run_w.sh > run_w.log 2>&1) || echo "!!!!! run_w FAILED (see exp_width/run_w.log)"
grep -E "VARYING WIDTHS|composition|EVAL_W" ../exp_width/run_w.log
echo "===== all finished $(date)"
