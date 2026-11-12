"""Step 10 (OPTIONAL extension, not the prescribed Part C method): voltage-aware parity checker.

The prescribed method (step 8) sums the 28 NAND2 leakages with every gate input at exactly 0 or
VDD. Inside the circuit, a NAND2 input driven by another NAND2 sits slightly off its rail, so the
receiving gate leaks differently. Here each NAND2 input driven by another NAND2 gets that driver's
ML-predicted output voltage (KCL-solved with the transistor models, no SPICE input), and the
receiving gate's nodes and leakage are solved with those real input voltages.
  depth 0  every input ideal (no voltage passing)
  depth 1  each driven input = its driver's ML output voltage (driver solved with ideal inputs)
Also: an ON transistor with |Vds| < 2 mV (below the training data's resolution and the clamp of the
ln(1-exp(-Vds/Vt)) feature) gets a linear-region channel current, the model's channel current at
|Vds| = 2 mV scaled by |Vds| / 2 mV (gate / body currents unchanged; KCL exact).
Loading between gates (the receiving gate's input current) is not modelled.
Main setup (varying widths). Output: results/vparity_summary.csv, results/vparity_depth<d>.csv.
Runtime ~3 h on 4 cores (depth 0: 4 NAND2 situations, depth 1: 16 more).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402
from itertools import product  # noqa: E402
from multiprocessing import get_context  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import final_method as F  # noqa: E402
from final_method import RES, C, G, M  # noqa: E402

DEPTHS = [int(x) for x in os.environ.get("DEPTHS", "0,1").split(",")]
MODELS, COR = None, None
GB = np.array([False, True, False, True])


def _eval_dev(ctx, dev, models, vd, vs):
    """Model currents (id, ig, is, ib) of one device at given drain / source voltages
    (real gate voltage and per-corner width, as final_method.device_currents)."""
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
    """final_method.device_currents with the small-Vds linear-region channel current (see top)."""
    name, t, d, gn, s, b, m = dev
    vd, vs = np.asarray(ctx.volt(d), float), np.asarray(ctx.volt(s), float)
    I = _eval_dev(ctx, dev, models, vd, vs)
    gate_hi = ctx.high(gn)
    on = gate_hi if t == "n" else ~gate_hi
    vds = np.abs(vd - vs)
    sel = on & (vds < M.VDS_CLAMP)
    if not sel.any():
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
    lv = F.logic_levels("NAND2", pat)
    init = {n: vdd * (0.5 if lv[n] is None else lv[n]) for n in ("x", "out")}
    ctx = CtxV("NAND2", pat, COR, init, {"a": va, "b": vb})
    for _ in range(F.SWEEPS):
        moved = 0.0
        for node in ("x", "out"):
            devs = [dv for dv in ctx.g["devices"] if node in dv[2:6]]
            lo, hi = np.zeros(ctx.n), vdd.copy()
            old = ctx.node_v[node].copy()
            for _ in range(F.ITERS):
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
        if moved < F.TOL:
            break
    # cut-set leakage (same rule as final_method.gate_leakage), with the real input voltages
    lctx = CtxV("NAND2", pat, COR, {n: ctx.node_v[n] for n in ("x", "out")}, {"a": va, "b": vb})
    total = np.zeros(lctx.n)
    for dev in lctx.g["devices"]:
        name, t, d, gn, s, b, m = dev
        I = G._device_currents(lctx, dev, MODELS)
        H = np.column_stack([lctx.high(d), lctx.high(gn), lctx.high(s), lctx.high(b)])
        same = H[:, 0] == H[:, 2]
        total += np.where(same, np.where(H[:, 0], (I * (~H & GB)).sum(1), -(I * (H & GB)).sum(1)), -(I * H).sum(1))
    return ctx.node_v["out"], total


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
    MODELS = F.load_models()
    COR = F.corners("varying").sort_values("corner").reset_index(drop=True)
    psp = pd.read_csv(RES / "circuit_spice_w.csv", dtype={"vector": str})
    psp = psp[psp.corner.isin(COR.corner)]
    vdd = COR["vdd"].to_numpy(float)
    memo, rows = {}, []
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
        m.to_csv(RES / f"vparity_depth{depth}.csv", index=False, float_format="%.6e")
        a = F.summary(m.assign(all=1), ["all"]).iloc[0]
        rows.append(dict(setup="varying", depth=depth, n=len(m), accuracy_pct=a.accuracy_pct,
                         median_err_pct=a.median_err_pct, p95_err_pct=a.p95_err_pct, bias_pct=a.bias_pct))
        print(f"PARITY (voltage-aware) depth {depth}: accuracy {a.accuracy_pct:.2f}%, median {a.median_err_pct:.2f}%, "
              f"p95 {a.p95_err_pct:.2f}%, bias {a.bias_pct:+.2f}% ({time.time() - t0:.0f} s)", flush=True)
        pd.DataFrame(rows).to_csv(RES / "vparity_summary.csv", index=False, float_format="%.4g")
    print("STEP 10 DONE", flush=True)


if __name__ == "__main__":
    main()
