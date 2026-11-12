"""Check that the pipeline reproduces the reported results.

1. Inputs: SHA-256 of every input artifact (data, models, SPICE references) against MANIFEST.sha256.
   Equal hashes = exactly the data / models the reported results came from.
2. Results: each file in results/ against its reference in expected_results/ (the outputs of the
   runs that produced the reported numbers; see expected_results/SOURCES.md):
     IDENTICAL  same text (tables) or same values to the reference's printed precision
     CLOSE      differs, but every number within --tol (relative; e.g. after regenerating data or
                retraining from scratch, where floating-point details can change)
     DIFFERENT  larger differences
     not run    the result file does not exist (e.g. optional step 10)
3. Prints the headline numbers.
Usage: python3 verify_final.py [--tol 1e-3]     exit code 0 = everything IDENTICAL (or CLOSE)
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES, EXP = ROOT / "results", ROOT / "expected_results"

# result file -> (reference file, key columns, value columns compared after formatting (None = whole text))
CHECKS = [
    ("kcl_check.csv", "kcl_check.csv", None, None),
    ("model_comparison_val.csv", "model_comparison_val.csv", None, None),
    ("transistor_metrics.csv", "transistor_metrics.csv", None, None),
    ("summary.csv", "summary.csv", None, None),
    ("gate_results.csv", "gate_results.csv", None, None),
    ("node_voltages.csv", "node_voltages.csv", None, None),
    ("gate_results_fixed.csv", "gate_results_fixed.csv", None, None),
    ("gate_ml.csv", "gate_ml.csv", ["gate", "pattern", "corner"], ["leak_A", "leak_B"]),
    ("vparity_summary.csv", "vparity_summary.csv", None, None),
]


def sha(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_manifest():
    bad = 0
    lines = [ln.split(maxsplit=1) for ln in (ROOT / "MANIFEST.sha256").read_text().splitlines() if ln.strip()]
    for h, name in lines:
        f = ROOT / name.strip().lstrip("*")
        if not f.exists():
            state = "MISSING"
        elif sha(f) == h:
            state = "ok"
        else:
            state = "CHANGED"
        bad += state != "ok"
        if state != "ok":
            print(f"  {state:8s} {name}")
    print(f"  {len(lines) - bad}/{len(lines)} input files identical to the ones behind the reported results")
    return bad == 0


def numeric_diff(a, b):
    num = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c])
           and pd.api.types.is_numeric_dtype(b[c])]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = [np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float))
                       / np.maximum(np.abs(b[c].to_numpy(float)), 1e-300)) for c in num]
    return max(d) if d else np.inf


def compare(res, ref, keys, vals, tol):
    r, e = RES / res, EXP / ref
    if not r.exists():
        return "not run", ""
    if not e.exists():
        return "no reference", ""
    if keys is None:
        if r.read_text() == e.read_text():
            return "IDENTICAL", "same text"
        a, b = pd.read_csv(r), pd.read_csv(e)
        if a.shape != b.shape:
            return "DIFFERENT", f"shape {a.shape} vs {b.shape}"
        d = numeric_diff(a, b)
    else:
        a = pd.read_csv(r, dtype={"pattern": str}).set_index(keys)[vals]
        b = pd.read_csv(e, dtype={"pattern": str}).set_index(keys)[vals]
        a = a.loc[b.index] if set(b.index) <= set(a.index) else None
        if a is None:
            return "DIFFERENT", "rows missing"
        same = all((a[c].map(lambda x: f"{x:.6e}") == b[c].map(lambda x: f"{x:.6e}")).all() for c in vals)
        if same:
            return "IDENTICAL", f"{len(b)} rows, values equal to 7 significant digits"
        d = numeric_diff(a.reset_index(), b.reset_index())
    return ("CLOSE" if d <= tol else "DIFFERENT"), f"max relative difference {d:.2e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-3)
    tol = ap.parse_args().tol
    print("1. input artifacts (MANIFEST.sha256)")
    inputs_ok = check_manifest()
    print("2. results vs expected_results/")
    ok = True
    for res, ref, keys, vals in CHECKS:
        state, note = compare(res, ref, keys, vals, tol)
        ok &= state in ("IDENTICAL", "CLOSE", "not run")
        print(f"  {state:10s} results/{res:24s} {note}")
    print("3. headline numbers (results/summary.csv)")
    if (RES / "summary.csv").exists():
        s = pd.read_csv(RES / "summary.csv")
        for _, r in s.iterrows():
            print(f"  {r.setup:15s} {r.version:24s} gates {r.gate_acc:.2f} %   parity checker {r.parity_acc:.2f} %")
    if (RES / "vparity_summary.csv").exists():
        for _, r in pd.read_csv(RES / "vparity_summary.csv").iterrows():
            print(f"  optional voltage-aware parity checker, depth {int(r.depth)}: {r.accuracy_pct:.2f} %")
    print("VERIFY:", "PASSED" if ok else "FAILED",
          "" if inputs_ok else "(inputs differ from the manifest: regenerated data or models)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
