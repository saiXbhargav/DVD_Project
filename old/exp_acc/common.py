"""Experiment exp_acc (accuracy steps): shared pieces.

Nothing outside exp_acc/ is modified; the project modules are only imported.
  GPBag         GP on ALL training rows: kernel hyper-parameters fitted once per
                output (on n_sub points, exactly as ml_utils.GPLite), then one exact
                GP per disjoint slice of <= slice_n rows with those hyper-parameters;
                prediction = linear mean (fitted on all rows) + average of the slice GPs.
  gate_leakage  circuits.ml_gate_leakage with a selectable side for a floating stack
                node (a node between two OFF devices): 'rule' = official rule,
                'flip' = the other side, 'avg' = mean of both. Both sides are exact
                by KCL; with model currents they differ by the node's KCL residual.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import circuits as G  # noqa: E402
import config as C  # noqa: E402
import ml_utils as M  # noqa: E402

DATA = HERE / "data"
OUT = HERE / "out"
for _d in (DATA, OUT, OUT / "models"):
    _d.mkdir(parents=True, exist_ok=True)


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


class GPCap(M.GPLite):
    """Only for unpickling the exp_extra GP_cap models (prediction = GPLite)."""


def _floating(ctx, k):
    devs = [dv for dv in ctx.g["devices"] if k in (dv[2], dv[4])]
    off = np.ones(ctx.n, bool) if devs else np.zeros(ctx.n, bool)
    for _, t, d, gn, s, b, m in devs:
        gh = ctx.high(gn)
        off &= ~gh if t == "n" else gh
    return off


def gate_leakage(gate, pattern, corners, node_v, models, mode="rule"):
    if mode == "avg":
        return 0.5 * (gate_leakage(gate, pattern, corners, node_v, models, "rule")
                      + gate_leakage(gate, pattern, corners, node_v, models, "flip"))
    ctx = G._Ctx(gate, pattern, corners, node_v)
    if mode == "flip":
        flips = {k: _floating(ctx, k) for k in ctx.node_v if k != "out"}
        for k, f in flips.items():
            ctx.hi_int[k] = np.where(f, ~ctx.hi_int[k], ctx.hi_int[k])
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


def summary(df, keys):
    rows = []
    for k, d in df.groupby(keys, sort=False):
        e = (d.leak_ml - d.leak_spice) / d.leak_spice
        rows.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)), n=len(d),
                         bias_pct=100 * e.median(), mean_abs_err_pct=100 * e.abs().mean(),
                         median_err_pct=100 * e.abs().median(), p95_err_pct=100 * e.abs().quantile(0.95),
                         accuracy_pct=100 - 100 * e.abs().mean()))
    return pd.DataFrame(rows)
