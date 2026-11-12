"""Final method: everything the final results need, in one module.

Transistor models (Part A)
  GPBag     averaged Gaussian process: kernel hyper-parameters fitted once per output on
            2,000 random points; one exact GP per disjoint slice of <= 9,700 training rows
            with those hyper-parameters; prediction = linear (ridge) mean + mean of the slices.
  TMX       ml_utils.TransistorModel (hi/lo frame, ln(|I|/W) target, KCL-derived current,
            two experts for OFF) with a chosen feature list.
  final models (GPX): ON cases  -> TMX(GPBag) with FEATS_X = official features + plain Vgs, Vgd, Vds
                      OFF cases -> ml_utils.TransistorModel(GPBag) with the official features
            trained on the 87,000-row pool: original training split (7,000) + 80,000 own rows.

Gates (Part B) and parity checker (Part C)
  * per-corner widths: every NMOS gets the corner's Wn, every PMOS its Wp (varying-width setup);
    a corner without wn / wp uses the fixed sizing of circuits.py (fixed-size reference)
  * real gate voltage: each transistor's Vg is the voltage of its gate node (input level, or
    the internal node that drives it in AND2 / OR2), clipped to [0, VDD]; Vb by the case rule
  * version A: internal / output node voltages from the gate's SPICE simulation
    version B: node voltages solved from KCL with the transistor models (no SPICE input)
  * leakage: cut-set rule (circuits.py docstring), stack-node side by the official rule
  * parity checker: prescribed method = sum of the 28 NAND2 leakages for each input vector

Numerics are copied unchanged from the experiment code that produced the reported results
(exp_acc/common.py, common2.py, common3.py, ml_nodes.py, exp_width/common_w.py,
exp_vg/common_vg.py, eval_vg.py, now in old/); verify_final.py checks bit-level agreement.
Importing this module installs device_currents() as circuits._device_currents.
"""
import glob
import os
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import circuits as G  # noqa: E402
import config as C  # noqa: E402
import ml_utils as M  # noqa: E402
import spice_utils as S  # noqa: E402

# ---------------------------------------------------------------- files
SMOKE = os.environ.get("SMOKE") == "1"          # quick test of steps 5-10: 3 corners, results/smoke/
DATA = C.DATA_DIR                               # data/
EXTRA = DATA / "train_extra"                    # data/train_extra/<case>_extra_0b.csv, <case>_more_0b.csv
RES = C.RESULTS_DIR / "smoke" if SMOKE else C.RESULTS_DIR      # results/
MODELS = C.MODEL_DIR                            # models_trained/<case>.joblib (final models)
FIG = C.FIG_DIR / "smoke" if SMOKE else C.FIG_DIR              # figures/
CORNERS_W = DATA / "corners_w.csv"              # 200 corners + wn, wp
for _d in (EXTRA, RES, MODELS, FIG):
    _d.mkdir(parents=True, exist_ok=True)

ON = ("nmos_on", "pmos_on")

# seeds (all disjoint): data 2026-2029, extra 5001-5304, more 7001-7304, split 42, models 7,
# corners 1234, widths 2468
SEED_EXTRA, SEED_MORE, BLOCKS, BLOCK_ROWS = 5000, 7000, 4, 10_000
SEED_W = 2468

# ---------------------------------------------------------------- ML node solver settings
ITERS, SWEEPS, TOL = 32, 8, 1e-7                # bisection steps, Gauss-Seidel sweeps, V


# ================================================================ transistor models
class GPBag(M.GPLite):
    def __init__(self, slice_n=9700, n_sub=2000, seed=C.SEED_MODEL):
        super().__init__(n_sub=n_sub, seed=seed, linear_mean=True)
        self.slice_n = slice_n

    def fit(self, X, Y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        self.sx = StandardScaler().fit(X)
        Xs = self.sx.transform(X)
        Y = np.asarray(Y).reshape(len(X), -1)
        rng = np.random.default_rng(self.seed)
        self.lin_, self.kernels_, self.slices_, self.noise_ = [], [], [], []
        for j in range(Y.shape[1]):
            ok = np.isfinite(Y[:, j])                         # NaN = below the noise floor
            Xj, yj = Xs[ok], Y[ok, j]
            lin = Ridge(alpha=1e-6).fit(Xj, yj)
            r = yj - lin.predict(Xj)
            idx = rng.choice(len(Xj), min(self.n_sub, len(Xj)), replace=False)
            kern = (ConstantKernel(1.0, (1e-2, 1e3)) * RBF(np.ones(X.shape[1]), (1e-2, 1e4))
                    + WhiteKernel(1e-6, (1e-10, 1e-2)))
            hp = GaussianProcessRegressor(kern, random_state=self.seed).fit(Xj[idx], r[idx])
            parts = np.array_split(rng.permutation(len(Xj)), int(np.ceil(len(Xj) / self.slice_n)))
            sl = []
            for p in parts:
                full = GaussianProcessRegressor(hp.kernel_, optimizer=None).fit(Xj[p], r[p])
                sl.append((Xj[p], full.alpha_.ravel()))
            self.lin_.append(lin)
            self.kernels_.append(hp.kernel_.k1)
            self.slices_.append(sl)
            self.noise_.append(hp.kernel_.k2.noise_level)
        return self

    def predict(self, X):
        Xs = self.sx.transform(X)
        out = np.empty((len(Xs), len(self.slices_)))
        for a in range(0, len(Xs), 2000):
            xb = Xs[a:a + 2000]
            for j, (k, sl, lin) in enumerate(zip(self.kernels_, self.slices_, self.lin_)):
                out[a:a + 2000, j] = np.mean([k(xb, Xt) @ al for Xt, al in sl], axis=0) + lin.predict(xb)
        return out


FEATS_X = M.PHYS + ["vgs", "vgd", "vds"]


class TMX(M.TransistorModel):
    def __init__(self, reg, case, feats=FEATS_X):
        super().__init__(reg, case)
        self.feats = list(feats)

    def _swap(self, fn, df):
        old = M.PHYS
        M.PHYS = self.feats
        try:
            return fn(df)
        finally:
            M.PHYS = old

    def fit(self, df):
        return self._swap(super().fit, df)

    def predict_frame(self, df):
        return self._swap(super().predict_frame, df)


def new_model(case):
    """Untrained final model of one case."""
    return TMX(GPBag(), case) if case in ON else M.TransistorModel(GPBag(), case)


def split(df):
    """70 / 15 / 15 train / validation / test split of the Part A data (seed 42)."""
    tr, rest = train_test_split(df, test_size=1 - C.SPLIT[0], random_state=C.SEED_SPLIT)
    va, te = train_test_split(rest, test_size=C.SPLIT[2] / (C.SPLIT[1] + C.SPLIT[2]), random_state=C.SEED_SPLIT)
    return tr, va, te


def extra_files(case):
    return (sorted(glob.glob(str(EXTRA / f"{case}_extra_*.csv")))
            + sorted(glob.glob(str(EXTRA / f"{case}_more_*.csv"))))


def training_pool(case):
    """Original training split + 8 x 10,000 own rows, in this order (87,000 rows)."""
    tr, _, _ = split(pd.read_csv(DATA / f"{case}.csv"))
    files = extra_files(case)
    assert len(files) == 2 * BLOCKS, files
    return pd.concat([tr] + [pd.read_csv(f) for f in files], ignore_index=True)


def load_models():
    import joblib
    return {c: joblib.load(MODELS / f"{c}.joblib") for c in C.CASES}


# ================================================================ corners
def corners(setup="varying"):
    """varying: 200 corners with per-corner wn / wp (main setup); fixed: the same 200 corners,
    fixed gate sizing (reference)."""
    cor = pd.read_csv(CORNERS_W if setup == "varying" else DATA / "corners.csv")
    return cor[cor.corner < 3] if SMOKE else cor


# ================================================================ netlists with per-corner widths
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


# ================================================================ ML device currents
def device_currents(ctx, dev, models):
    """Terminal currents (n x 4: id, ig, is, ib; > 0 = out of the device) of one transistor at
    every corner, with its REAL gate voltage and its width (corner wn / wp if present)."""
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


G._device_currents = device_currents            # every evaluator below uses it


# ================================================================ gate leakage (cut-set rule)
def gate_leakage(gate, pattern, corners, node_v, models):
    ctx = G._Ctx(gate, pattern, corners, node_v)
    high = ctx.high
    total = np.zeros(ctx.n)
    gb = np.array([False, True, False, True])
    for dev in ctx.g["devices"]:
        name, t, d, gn, s, b, m = dev
        I = G._device_currents(ctx, dev, models)
        H = np.column_stack([high(d), high(gn), high(s), high(b)])
        same = H[:, 0] == H[:, 2]
        c_same_hi = (I * (~H & gb)).sum(1)
        c_same_lo = -(I * (H & gb)).sum(1)
        c_cross = -(I * H).sum(1)
        total += np.where(same, np.where(H[:, 0], c_same_hi, c_same_lo), c_cross)
    return total


# ================================================================ version B: ML node voltages
def logic_levels(gate, pat):
    """Logic value (0 / 1) of every node, None for a stack node between two OFF devices."""
    a = int(pat[0])
    b = int(pat[1]) if len(pat) > 1 else None
    if gate == "INV":
        return {"out": 1 - a}
    if gate in ("NAND2", "AND2"):
        n = 1 - (a & b)
        x = 0 if b == 1 else (n if a == 1 else None)     # MNB (gate b) to ground, MNA (gate a) to the output
        return {"x": x, "out": n} if gate == "NAND2" else {"x": x, "nd": n, "out": 1 - n}
    n = 1 - (a | b)
    y = 1 if a == 0 else (n if b == 0 else None)         # MPA (gate a) to VDD, MPB (gate b) to the output
    return {"y": y, "out": n} if gate == "NOR2" else {"y": y, "nr": n, "out": 1 - n}


def solve_nodes(gate, pat, cs, models):
    """Every node starts at its logic level (VDD/2 for a floating stack node); each node voltage is
    the KCL root of the model currents into it (bisection on [0, VDD], ITERS steps), Gauss-Seidel
    over the nodes until no node moves by more than TOL. Returns ({node: V}, sweeps)."""
    vdd = cs["vdd"].to_numpy(float)
    lv = logic_levels(gate, pat)
    nodes = G.GATES[gate]["nodes"]
    init = {n: vdd * (0.5 if lv[n] is None else lv[n]) for n in nodes}
    ctx = G._Ctx(gate, pat, cs, init)                    # logic sides come from the logic levels
    for sweep in range(1, SWEEPS + 1):
        moved = 0.0
        for node in nodes:
            devs = [dv for dv in ctx.g["devices"] if node in dv[2:6]]
            lo, hi = np.zeros(ctx.n), vdd.copy()
            old = ctx.node_v[node].copy()
            for _ in range(ITERS):
                mid = 0.5 * (lo + hi)
                ctx.node_v[node] = mid
                f = np.zeros(ctx.n)                      # model current flowing out of the devices into the node
                for dv in devs:
                    I = G._device_currents(ctx, dv, models)
                    for col, term in enumerate(dv[2:6]):
                        if term == node:
                            f += I[:, col]
                lo = np.where(f > 0, mid, lo)
                hi = np.where(f > 0, hi, mid)
            ctx.node_v[node] = 0.5 * (lo + hi)
            moved = max(moved, float(np.max(np.abs(ctx.node_v[node] - old))))
        if moved < TOL:
            break
    return {n: ctx.node_v[n] for n in nodes}, sweep


# ================================================================ metrics and parity sum
PATS = ["00", "01", "10", "11"]


def summary(df, keys):
    """accuracy = 100 - mean |relative error| (%); also median, p95 and bias (median signed error)."""
    rows = []
    for k, d in df.groupby(keys, sort=False):
        e = (d.leak_ml - d.leak_spice) / d.leak_spice
        rows.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)), n=len(d),
                         bias_pct=100 * e.median(), mean_abs_err_pct=100 * e.abs().mean(),
                         median_err_pct=100 * e.abs().median(), p95_err_pct=100 * e.abs().quantile(0.95),
                         accuracy_pct=100 - 100 * e.abs().mean()))
    return pd.DataFrame(rows)


def vec_counts():
    """256 input vectors x 4 NAND2 patterns: how many of the 28 NAND2 see each pattern."""
    vectors = ["".join(v) for v in product("01", repeat=8)]
    return pd.DataFrame({v: pd.Series([p for _, p in G.parity_logic(v)[1]]).value_counts()
                         .reindex(PATS, fill_value=0) for v in vectors}).T


def parity_sum(nand, cnt, psp):
    """nand: corner x pattern NAND2 leakages -> parity leakage per (corner, vector), merged with SPICE."""
    ml = pd.DataFrame(nand[PATS].to_numpy() @ cnt.to_numpy().T, index=nand.index, columns=cnt.index)
    ml = ml.stack().rename("leak_ml").reset_index()
    ml.columns = ["corner", "vector", "leak_ml"]
    return psp.merge(ml, on=["corner", "vector"])


def transistor_metrics(model, d):
    """Per-current errors of one model on one data split (val / test)."""
    P, Y = model.predict(d), d[C.CURRENTS].to_numpy()
    e = 100 * np.abs(P - Y) / np.abs(Y)
    big = np.abs(Y) >= 1e-11
    lp, ly = np.log10(np.abs(P) + 1e-30), np.log10(np.abs(Y) + 1e-30)
    ok = np.abs(Y) > 1e-16
    r2 = [1 - ((lp[ok[:, j], j] - ly[ok[:, j], j]) ** 2).sum() / ((ly[ok[:, j], j] - ly[ok[:, j], j].mean()) ** 2).sum()
          for j in range(4)]
    row = {}
    for j, c in enumerate(C.CURRENTS):
        row[f"{c}_median_pct"] = np.median(e[:, j])
        row[f"{c}_p95_pct"] = np.quantile(e[np.isfinite(e[:, j]), j], .95)
        row[f"{c}_r2_log"] = r2[j]
    row["mean_of_medians_pct"] = np.mean([np.median(e[:, j]) for j in range(4)])
    row["ge1e11_median_pct"] = np.median(e[big])
    row["ge1e11_p95_pct"] = np.quantile(e[big], .95)
    return row, P
