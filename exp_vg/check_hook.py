"""Sanity check of the real-gate-voltage code path: current final models (GPX), fixed sizes,
version A. Expected close to 98.65 % (internal nodes are only mV from their rails there)."""
import joblib

import eval_vg
from common_vg import ON, C, OUT

d = OUT.parent.parent / "exp_acc" / "out" / "models"
gpx = {c: joblib.load(d / f"{'GPBag87X' if c in ON else 'GPBag87'}_{c}.joblib") for c in C.CASES}
eval_vg.SUF = "_hookcheck"
print(eval_vg.fixed_sizes(gpx))
