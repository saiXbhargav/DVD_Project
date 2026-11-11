#!/bin/bash
# Overnight queue: waits for round 2 (run2.sh) to finish, then runs the fresh check and the
# dataset-size study. Each job runs even if the previous one failed. Log: exp_acc/overnight.log
cd "$(dirname "$0")" || exit 1
echo "===== waiting for run2.sh $(date)"
while pgrep -f "run2.sh" > /dev/null; do sleep 60; done
echo "===== fresh_check $(date)"
python3 -u fresh_check.py || echo "!!!!! fresh_check FAILED"
echo "===== size_study $(date)"
python3 -u size_study.py || echo "!!!!! size_study FAILED"
echo "===== overnight finished $(date)"
