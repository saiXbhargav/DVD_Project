"""Fix 1 (exp_vg/): gate-voltage-aware transistor models. Shared pieces; new files only.

Problem: the transistor models were trained with the gate exactly at 0 or VDD (the document's
case definition), but in a two-stage gate (AND2 = NAND2 + INV, OR2 = NOR2 + INV) the INV stage's
gate is the internal node nd / nr, which with varying widths sags up to ~90 mV from its rail.
Fix: extra training rows with the gate slightly off its rail (VDD - delta or 0 + delta), and at
gate level each transistor gets its real gate voltage (the node voltage) instead of the rule.

Reads (never writes) exp_acc/ (87k-row pool, GP code) and exp_width/ (varying-width corners and
SPICE). Everything new goes to exp_vg/data and exp_vg/out.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "exp_acc", ROOT / "exp_width"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
import circuits as G  # noqa: E402
import common  # noqa: E402,F401  (GPBag; also needed to unpickle)
import common2  # noqa: E402,F401  (TMX, pool87)
import common_w  # noqa: E402  (varying-width corners / SPICE; installs its own hook, replaced below)

DATA, OUT = HERE / "data", HERE / "out"
for _d in (DATA, OUT, OUT / "models"):
    _d.mkdir(parents=True, exist_ok=True)
SMOKE = os.environ.get("SMOKE") == "1"
ON = ("nmos_on", "pmos_on")
DELTA_MIN, DELTA_MAX = 1e-4, 0.15      # V, gate offset from its rail (log-uniform)


def device_currents_vg(ctx, dev, models):
    """Terminal currents with each transistor's REAL gate voltage (its gate node's voltage) and
    width (corner wn / wp when the corner has them, otherwise the fixed gate sizing)."""
    name, t, d, gn, s, b, m = dev
    n, vdd, corners = ctx.n, ctx.vdd, ctx.corners
    gate_hi = ctx.high(gn)
    on = gate_hi if t == "n" else ~gate_hi
    if "wn" in corners.columns:
        width = corners["wn" if t == "n" else "wp"].to_numpy(float)
    else:
        width = np.full(n, m * G.WU)
    vg_all = np.clip(ctx.volt(gn), 0.0, vdd)
    I = np.zeros((n, 4))
    for state in (True, False):
        sel = on == state
        if not sel.any():
            continue
        case = f"{'nmos' if t == 'n' else 'pmos'}_{'on' if state else 'off'}"
        _, _, vb_rule = C.CASES[case]
        cs = corners[sel]
        vsel = vdd[sel]
        df = pd.DataFrame({
            "vdd": vsel, "temp": cs["temp"].to_numpy(float), "w": width[sel],
            "vd": ctx.volt(d)[sel], "vs": ctx.volt(s)[sel], "vg": vg_all[sel],
            "vb": vsel if vb_rule == "vdd" else np.full(sel.sum(), vb_rule),
        })
        for p in C.PROC_PARAMS:
            df[p] = cs[f"{t}_{p}"].to_numpy(float)
        I[sel] = models[case].predict(df)
    return I


G._device_currents = device_currents_vg     # this process only
