"""Varying-width evaluation (exp_width/): shared pieces. New files only; nothing else is modified.

Width design: each corner keeps its PVT values (the official 200 corners) and gets one random
NMOS width Wn and one random PMOS width Wp, uniform in W_RANGE = 16-128 nm (1-8 x Lmin, the
project document's width range and the trained range). All NMOS of a gate / circuit share Wn,
all PMOS share Wp (the same way the process values are shared).

The ML side reuses the exp_acc code unchanged: circuits._device_currents is replaced (in this
process only) by a version that takes each transistor's width from the corner's wn / wp.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "exp_acc"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import circuits as G  # noqa: E402
import config as C  # noqa: E402
import spice_utils as S  # noqa: E402
import common  # noqa: E402,F401  (exp_acc: GPBag, needed to unpickle the models)
import common2  # noqa: E402,F401 (exp_acc: TMX, needed to unpickle the models)

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
SEED_W = 2468                       # widths of the 200 official corners
SMOKE = os.environ.get("SMOKE") == "1"
ON = ("nmos_on", "pmos_on")


def load_models():
    """Final GPX models: GPBag87X for the ON cases, GPBag87 for the OFF cases."""
    d = HERE.parent / "exp_acc" / "out" / "models"
    return {c: joblib.load(d / f"{'GPBag87X' if c in ON else 'GPBag87'}_{c}.joblib") for c in C.CASES}


def corners_w():
    f = OUT / "corners_w.csv"
    if not f.exists():
        base = G.load_corners()
        rng = np.random.default_rng(SEED_W)
        base["wn"] = rng.uniform(*C.W_RANGE, len(base))
        base["wp"] = rng.uniform(*C.W_RANGE, len(base))
        base.to_csv(f, index=False, float_format="%.10e")
    cor = pd.read_csv(f)
    return cor[cor.corner < 3] if SMOKE else cor


def device_lines_w(devs, wn, wp, prefix=""):
    model = {"n": "nmos", "p": "pmos"}
    return [f"{prefix}{name} {d} {g} {s} {b} {model[t]} W={S.fmt(wn if t == 'n' else wp)} L={S.fmt(C.LMIN)}"
            for name, t, d, g, s, b, m in devs]


def gate_netlist_w(gate, row):
    g = G.GATES[gate]
    lines = G.header(row, f"{gate} corner {int(row['corner'])} Wn={row['wn']:.4g} Wp={row['wp']:.4g}")
    lines += [f"V{i} {i} 0 0" for i in g["inputs"]]
    lines += device_lines_w(g["devices"], row["wn"], row["wp"])
    lines += [".op", ".end"]
    return "\n".join(lines) + "\n"


def parity_netlist_w(row):
    lines = G.header(row, f"8-bit parity checker (28 NAND2) corner {int(row['corner'])} "
                          f"Wn={row['wn']:.4g} Wp={row['wp']:.4g}")
    lines += [f"V{d} {d} 0 0" for d in G.PARITY_INPUTS]
    for inst, a, b, out in G.parity_structure():
        devs = [(f"{n}_{inst}", t, *[{"a": a, "b": b, "out": out, "x": f"{inst}_x"}.get(x, x)
                                     for x in (dn, g, s, bb)], m)
                for n, t, dn, g, s, bb, m in G._NAND2]
        lines += device_lines_w(devs, row["wn"], row["wp"])
    lines += [".op", ".end"]
    return "\n".join(lines) + "\n"


def device_currents_w(ctx, dev, models):
    """circuits._device_currents with the width taken from the corner (wn for NMOS, wp for PMOS)."""
    name, t, d, gn, s, b, m = dev
    n, vdd, corners = ctx.n, ctx.vdd, ctx.corners
    gate_hi = ctx.high(gn)
    on = gate_hi if t == "n" else ~gate_hi
    width = corners["wn" if t == "n" else "wp"].to_numpy(float)
    I = np.zeros((n, 4))
    for state in (True, False):
        sel = on == state
        if not sel.any():
            continue
        case = f"{'nmos' if t == 'n' else 'pmos'}_{'on' if state else 'off'}"
        _, vg_rule, vb_rule = C.CASES[case]
        cs = corners[sel]
        vsel = vdd[sel]
        df = pd.DataFrame({
            "vdd": vsel, "temp": cs["temp"].to_numpy(float), "w": width[sel],
            "vd": ctx.volt(d)[sel], "vs": ctx.volt(s)[sel],
            "vg": vsel if vg_rule == "vdd" else np.full(sel.sum(), vg_rule),
            "vb": vsel if vb_rule == "vdd" else np.full(sel.sum(), vb_rule),
        })
        for p in C.PROC_PARAMS:
            df[p] = cs[f"{t}_{p}"].to_numpy(float)
        I[sel] = models[case].predict(df)
    return I


G._device_currents = device_currents_w      # this process only: every evaluator now uses per-corner widths
