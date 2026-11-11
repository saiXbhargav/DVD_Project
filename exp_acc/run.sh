#!/bin/bash
# exp_acc: run all steps in order (resumable); log in exp_acc/run.log
cd "$(dirname "$0")" || exit 1
set -e
for s in gen_data gate_val_spice train evaluate; do
    echo "===== $s $(date)"
    python3 -u "$s.py"
done
echo "===== finished $(date)"
