"""Voltage-aware parity checker (improvement over the prescribed sum; new files only).

The prescribed Part C method sums the 28 NAND2 gate leakages with every gate input taken at
exactly 0 or VDD. Inside the circuit, a NAND2 input driven by another NAND2 sits slightly off
its rail (a logic 1 driven through a leaking stack sags below VDD), so the receiving gate leaks
more. Here each NAND2 input driven by another NAND2 gets that driver's ML-predicted output
voltage (node voltages solved from KCL with the four transistor models, no SPICE input), and the
receiving gate's internal nodes and leakage are solved with those real input voltages.

DEPTH d = how many gates back the voltages are followed:
  d = 0  every input ideal (= the prescribed sum with ML node voltages; consistency check)
  d = 1  each input = its driver's output, driver solved with ideal inputs
  d = 2  the driver's own driven inputs are real as well
Loading between gates (the receiving gate's input current) is not modelled.
Models: current final models (GPX) with real gate voltages and per-corner widths (exp_vg hook).
SETUP=varying (default: exp_width corners and SPICE) or SETUP=fixed (official corners / sizes).
Output: exp_vparity/out/vparity_<setup>_summary.csv and per-vector predictions.
SMOKE=1 -> 3 corners.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
import time  # noqa: E402
from itertools import product  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp_vg"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common_vg import C, G, ON  # noqa: E402  (installs the real-gate-voltage + width hook)
import common_w  # noqa: E402
import ml_nodes  # noqa: E402
from common import summary  # noqa: E402

import ml_utils as M  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
SETUP = os.environ.get("SETUP", "varying")
LINVDS = os.environ.get("LINVDS", "1") == "1"   # linear-region channel current for ON devices below 2 mV


def _eval_dev(ctx, dev, models, vd, vs):
    """Model currents (id, ig, is, ib) of one device at given drain / source voltages
    (real gate voltage and per-corner width, as in the exp_vg hook)."""
    name, t, d, gn, s, b, m = dev
    n, vdd, corners = ctx.n, ctx.vdd, ctx.corners
    gate_hi = ctx.high(gn)
    on = gate_hi if t == "n" else ~gate_hi
    width = corners["wn" if t == "n" else "wp"].to_numpy(float) if "wn" in corners.columns else np.full(n, m * G.WU)
    vg_all = np.clip(ctx.volt(gn), 0.0, vdd)
    I = np.zeros((n, 4))
    for state in (True, False):
        sel = on == state
        if not sel.any():
            continue
        case = f"{'nmos' if t == 'n' else 'pmos'}_{'on' if state else 'off'}"
        vb_rule = C.CASES[case][2]
        cs = corners[sel]
        df = pd.DataFrame({"vdd": vdd[sel], "temp": cs["temp"].to_numpy(float), "w": width[sel],
                           "vd": vd[sel], "vs": vs[sel], "vg": vg_all[sel],
                           "vb": vdd[sel] if vb_rule == "vdd" else np.full(sel.sum(), vb_rule)})
        for p in C.PROC_PARAMS:
            df[p] = cs[f"{t}_{p}"].to_numpy(float)
        I[sel] = models[case].predict(df)
    return I


def dev_currents_lin(ctx, dev, models):
    """As the exp_vg hook, but an ON transistor with |Vds| < 2 mV (below the training data's
    resolution and the clamp of the ln(1-exp(-Vds/Vt)) feature) gets a linear-region channel
    current: the model's channel current at |Vds| = 2 mV scaled by |Vds| / 2 mV. Gate and body
    currents stay as the model predicts; KCL holds exactly."""
    name, t, d, gn, s, b, m = dev
    vd, vs = np.asarray(ctx.volt(d), float), np.asarray(ctx.volt(s), float)
    I = _eval_dev(ctx, dev, models, vd, vs)
    gate_hi = ctx.high(gn)
    on = gate_hi if t == "n" else ~gate_hi
    vds = np.abs(vd - vs)
    sel = on & (vds < M.VDS_CLAMP)
    if not LINVDS or not sel.any():
        return I
    hi_d = vd >= vs
    vhi, vlo = np.maximum(vd, vs), np.minimum(vd, vs)
    up = vlo + M.VDS_CLAMP <= ctx.vdd                 # move the high terminal up, or the low one down
    vhi2 = np.where(up, vlo + M.VDS_CLAMP, vhi)
    vlo2 = np.where(up, vlo, vhi - M.VDS_CLAMP)
    I0 = _eval_dev(ctx, dev, models, np.where(hi_d, vhi2, vlo2), np.where(hi_d, vlo2, vhi2))
    hi0, lo0 = np.where(hi_d, I0[:, 0], I0[:, 2]), np.where(hi_d, I0[:, 2], I0[:, 0])
    hia, loa = np.where(hi_d, I[:, 0], I[:, 2]), np.where(hi_d, I[:, 2], I[:, 0])
    ich = 0.5 * (lo0 - hi0) * vds / M.VDS_CLAMP       # channel current, high -> low terminal
    avg = 0.5 * (hia + loa)                            # = -(ig + ib) / 2 at the actual point
    new_hi, new_lo = avg - ich, avg + ich
    I[sel, 0] = np.where(hi_d, new_hi, new_lo)[sel]
    I[sel, 2] = np.where(hi_d, new_lo, new_hi)[sel]
    return I


G._device_currents = dev_currents_lin            # this process only
DEPTHS = [int(x) for x in os.environ.get("DEPTHS", "0,1,2").split(",")]
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
MODELS, COR = None, None
GB = np.array([False, True, False, True])


class CtxV(G._Ctx):
    """Gate context whose inputs sit at given voltages (logic levels unchanged)."""

    def __init__(self, gate, pattern, corners, node_v, vin):
        self.vin = vin
        super().__init__(gate, pattern, corners, node_v)

    def volt(self, node):
        if node in self.vin:
            return self.vin[node]
        return super().volt(node)


def solve_nand(pat, va, vb):
    """ML node voltages (x, out) and leakage of a NAND2 whose inputs sit at va, vb (arrays)."""
    vdd = COR["vdd"].to_numpy(float)
    lv = ml_nodes.logic_levels("NAND2", pat)
    init = {n: vdd * (0.5 if lv[n] is None else lv[n]) for n in ("x", "out")}
    ctx = CtxV("NAND2", pat, COR, init, {"a": va, "b": vb})
    for _ in range(ml_nodes.SWEEPS):
        moved = 0.0
        for node in ("x", "out"):
            devs = [dv for dv in ctx.g["devices"] if node in dv[2:6]]
            lo, hi = np.zeros(ctx.n), vdd.copy()
            old = ctx.node_v[node].copy()
            for _ in range(ml_nodes.ITERS):
                mid = 0.5 * (lo + hi)
                ctx.node_v[node] = mid
                f = np.zeros(ctx.n)
                for dv in devs:
                    I = G._device_currents(ctx, dv, MODELS)
                    for col, term in enumerate(dv[2:6]):
                        if term == node:
                            f += I[:, col]
                lo, hi = np.where(f > 0, mid, lo), np.where(f > 0, hi, mid)
            ctx.node_v[node] = 0.5 * (lo + hi)
            moved = max(moved, float(np.max(np.abs(ctx.node_v[node] - old))))
        if moved < ml_nodes.TOL:
            break
    # cut-set leakage (same rule as common.gate_leakage, mode "rule"), with the real input voltages
    lctx = CtxV("NAND2", pat, COR, {n: ctx.node_v[n] for n in ("x", "out")}, {"a": va, "b": vb})
    total = np.zeros(lctx.n)
    for dev in lctx.g["devices"]:
        name, t, d, gn, s, b, m = dev
        I = G._device_currents(lctx, dev, MODELS)
        H = np.column_stack([lctx.high(d), lctx.high(gn), lctx.high(s), lctx.high(b)])
        same = H[:, 0] == H[:, 2]
        total += np.where(same, np.where(H[:, 0], (I * (~H & GB)).sum(1), -(I * (H & GB)).sum(1)), -(I * H).sum(1))
    return ctx.node_v["out"], total


# ---------------------------------------------------------------- source keys
# ("in", bit): a primary input; ("g", pattern, keyA, keyB): the output of a NAND2 whose inputs are keyA, keyB
def logic(k):
    return k[1] if k[0] == "in" else 1 - (logic(k[2]) & logic(k[3]))


def trunc(k, d):
    if k[0] == "in":
        return k
    if d <= 0:
        return ("g", k[1], ("in", logic(k[2])), ("in", logic(k[3])))
    return ("g", k[1], trunc(k[2], d - 1), trunc(k[3], d - 1))


def level(k):
    return 0 if k[0] == "in" else 1 + max(level(k[2]), level(k[3]))


def instance_keys(depth):
    """For every vector: the (keyA, keyB) of each of the 28 NAND2, inputs followed `depth` gates back."""
    vectors = ["".join(v) for v in product("01", repeat=8)]
    out = {}
    for vec in vectors:
        src = {d: ("in", int(b)) for d, b in zip(G.PARITY_INPUTS, vec)}
        keys = []
        for inst, a, b, o in G.parity_structure():
            ka = src[a] if depth > 0 or src[a][0] == "in" else ("in", logic(src[a]))
            kb = src[b] if depth > 0 or src[b][0] == "in" else ("in", logic(src[b]))
            if depth > 0:
                ka, kb = trunc(ka, depth - 1), trunc(kb, depth - 1)
            keys.append((ka, kb))
            pa, pb = logic(src[a]), logic(src[b])
            src[o] = ("g", f"{pa}{pb}", src[a], src[b])
        out[vec] = keys
    return out


def _task(args):
    pair, va, vb = args
    pat = f"{logic(pair[0])}{logic(pair[1])}"
    t0 = time.time()
    vout, leak = solve_nand(pat, va, vb)
    return pair, vout, leak, time.time() - t0


def run_pairs(pairs, memo, vdd):
    """Solve all (keyA, keyB) pairs not yet in memo, lowest level first (drivers before receivers)."""
    todo = sorted({p for p in pairs if p not in memo}, key=lambda p: max(level(p[0]), level(p[1])))
    need = set()

    def add(k):                                      # drivers of driven inputs must be solved too
        if k[0] == "g":
            for kk in (k[2], k[3]):
                add(kk)
            need.add((k[2], k[3]))
    for p in todo:
        add(p[0])
        add(p[1])
    todo = sorted(set(todo) | {p for p in need if p not in memo}, key=lambda p: max(level(p[0]), level(p[1])))
    by_level = {}
    for p in todo:
        by_level.setdefault(max(level(p[0]), level(p[1])), []).append(p)

    def volt(k):
        return vdd * k[1] if k[0] == "in" else memo[(k[2], k[3])][0]
    with get_context("fork").Pool(4) as pool:
        for lv in sorted(by_level):
            jobs = [(p, volt(p[0]), volt(p[1])) for p in by_level[lv]]
            for pair, vout, leak, dt in pool.imap_unordered(_task, jobs):
                memo[pair] = (vout, leak)
            print(f"    level {lv}: {len(jobs)} NAND2 situations solved", flush=True)


def main():
    global MODELS, COR
    d = HERE.parent / "exp_acc" / "out" / "models"
    MODELS = {c: joblib.load(d / f"{'GPBag87X' if c in ON else 'GPBag87'}_{c}.joblib") for c in C.CASES}
    if SETUP == "varying":
        COR = common_w.corners_w()
        psp = pd.read_csv(common_w.OUT / "circuit_spice_w.csv", dtype={"vector": str})
    else:
        COR = G.load_corners()
        psp = pd.read_csv(C.RESULTS_DIR / "circuit_spice.csv", dtype={"vector": str})
    if SMOKE:
        COR = COR[COR.corner < 3]
    COR = COR.sort_values("corner").reset_index(drop=True)
    psp = psp[psp.corner.isin(COR.corner)]
    vdd = COR["vdd"].to_numpy(float)
    memo, rows = {}, []
    tag = f"{SETUP}{'_smoke' if SMOKE else ''}"
    for depth in DEPTHS:
        t0 = time.time()
        keys = instance_keys(depth)
        pairs = {p for ks in keys.values() for p in ks}
        print(f"depth {depth}: {len(pairs)} distinct NAND2 situations", flush=True)
        run_pairs(pairs, memo, vdd)
        pred = []
        for vec, ks in keys.items():
            tot = np.sum([memo[p][1] for p in ks], axis=0)
            pred.append(pd.DataFrame({"corner": COR.corner.to_numpy(), "vector": vec, "leak_ml": tot}))
        pred = pd.concat(pred, ignore_index=True)
        m = psp.merge(pred, on=["corner", "vector"])
        m.to_csv(OUT / f"vparity_{tag}_depth{depth}.csv", index=False, float_format="%.6e")
        a = summary(m.assign(all=1), ["all"]).iloc[0]
        rows.append(dict(setup=SETUP, depth=depth, n=len(m), accuracy_pct=a.accuracy_pct,
                         median_err_pct=a.median_err_pct, p95_err_pct=a.p95_err_pct, bias_pct=a.bias_pct))
        print(f"PARITY {SETUP} depth {depth}: accuracy {a.accuracy_pct:.2f}%, median {a.median_err_pct:.2f}%, "
              f"p95 {a.p95_err_pct:.2f}%, bias {a.bias_pct:+.2f}% ({time.time() - t0:.0f} s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"vparity_{tag}_summary.csv", index=False, float_format="%.4g")
    print("VPARITY DONE", flush=True)


if __name__ == "__main__":
    main()
