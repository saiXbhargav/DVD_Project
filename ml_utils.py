"""Feature engineering, target transform, model zoo and metrics for the transistor models.

1. Symmetric frame. A MOSFET is source/drain symmetric, so each sample is mapped to
   a frame where terminal "hi" is the higher of (Vd, Vs) and "lo" the lower. The
   model predicts (I_hi, Ig, I_lo, Ib) and these are mapped back to (Id, Ig, Is, Ib).
2. Sign handling. In this frame almost every current has a fixed sign for a given
   case. Each fixed-sign current is modelled as y = ln(|I| / W) (current per unit
   width, log scale, so an error of 0.01 in y is ~1 % in I). At most one current
   per case changes sign (e.g. I_lo of NMOS OFF, where sub-threshold current and
   junction / gate leakage compete); that one is not learned but derived from
   Kirchhoff's current law: I_k = -(sum of the other three). The predicted
   currents therefore satisfy KCL exactly.
3. Physics features. Thermal voltage scaled biases (Vgs/Vt, Vds/Vt, ...) and
   ln(1 - exp(-Vds/Vt)) give the sub-threshold dependence directly.
4. Noise floor. ngspice resolves these currents only in steps of ~2e-16 A, so
   values below NOISE_FLOOR = 3e-15 A are partly numerical noise and their log is
   not learnable. The GP ignores such samples for that output; the other model
   types clip them at the floor. (They are ~1e8 x smaller than gate leakage.)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import config as C

K_B_Q = 8.617333262e-5          # k/q in V/K
NOISE_FLOOR = 3e-15             # A, ~14 x the ngspice current resolution here
VDS_CLAMP = 2e-3                # V, lower clamp of Vds inside the ln(1-exp(-Vds/Vt)) feature
FRAME = ["i_hi", "ig", "i_lo", "ib"]


def features(df):
    """Physics-informed features in the symmetric (hi/lo) frame."""
    vd, vs, vg, vb = (df[k].to_numpy(float) for k in ("vd", "vs", "vg", "vb"))
    vhi, vlo = np.maximum(vd, vs), np.minimum(vd, vs)
    tk = df["temp"].to_numpy(float) + 273.15
    vt = K_B_Q * tk
    vds = vhi - vlo
    vth = np.abs(df["vth0"].to_numpy(float))
    X = pd.DataFrame({
        "vdd": df["vdd"].to_numpy(float), "vhi": vhi, "vlo": vlo, "vds": vds,
        "vgs": vg - vlo, "vgd": vg - vhi, "vbs": vb - vlo, "vbd": vb - vhi,
        "temp": df["temp"].to_numpy(float), "inv_vt": 1.0 / vt,
        "vds_vt": vds / vt, "vgs_vt": (vg - vlo) / vt, "vgd_vt": (vg - vhi) / vt,
        "vbs_vt": (vb - vlo) / vt, "vth_vt": vth / vt,
        # clamped at Vds = 2 mV: ln(1-exp(-Vds/Vt)) -> -inf as Vds -> 0, and ON devices
        # in a gate sit at Vds ~ uV-mV, far below the training data (vds_vt still
        # carries the exact small Vds, linearly)
        "ln1m": np.log(-np.expm1(-np.maximum(vds, VDS_CLAMP) / vt)),
        "logw": np.log(df["w"].to_numpy(float)),
        "vth0": vth, "toxe": df["toxe"].to_numpy(float) * 1e9,
        "ndep": df["ndep"].to_numpy(float) * 1e-18, "u0": df["u0"].to_numpy(float),
        "xj": df["xj"].to_numpy(float) * 1e9,
    })
    return X


def to_frame(df):
    """Terminal currents -> symmetric-frame currents (I_hi, Ig, I_lo, Ib)."""
    hi_is_d = (df["vd"] >= df["vs"]).to_numpy()
    ihi = np.where(hi_is_d, df["id"], df["is"])
    ilo = np.where(hi_is_d, df["is"], df["id"])
    return np.column_stack([ihi, df["ig"], ilo, df["ib"]])


def from_frame(I, df):
    """Symmetric-frame currents -> terminal currents (Id, Ig, Is, Ib)."""
    hi_is_d = (df["vd"] >= df["vs"]).to_numpy()
    idr = np.where(hi_is_d, I[:, 0], I[:, 2])
    isr = np.where(hi_is_d, I[:, 2], I[:, 0])
    return np.column_stack([idr, I[:, 1], isr, I[:, 3]])


def rel_err(pred, true):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(pred - true) / np.abs(true)


# ---------------------------------------------------------------- model zoo
# Model inputs: thermal-voltage-scaled biases + process parameters. vg and vb are fixed
# per case (0 or VDD), so (vdd, vgs, vgd) fully determine the terminal voltages.
PHYS = ["vdd", "vgs_vt", "vgd_vt", "vds_vt", "ln1m", "inv_vt", "logw", "vth_vt",
        "toxe", "ndep", "u0", "xj"]


class GPLite:
    """Exact Gaussian-process regression (ARD RBF kernel), one GP per output.

    Kernel hyper-parameters are optimised on a random subset (n_sub points; the
    O(n^3) optimisation is too slow on all 7000), then an exact GP with those
    hyper-parameters is conditioned on all training points. Only what the mean
    prediction needs is kept (training inputs, kernel, weights alpha), so the
    saved model is small: mean(x*) = k(x*, X_train) @ alpha.
    """

    def __init__(self, n_sub=2000, seed=C.SEED_MODEL, linear_mean=True):
        self.n_sub, self.seed, self.linear_mean = n_sub, seed, linear_mean

    def fit(self, X, Y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        self.sx = StandardScaler().fit(X)
        Xs = self.sx.transform(X)
        Y = np.asarray(Y).reshape(len(X), -1)
        rng = np.random.default_rng(self.seed)
        self.Xs_, self.kernels_, self.alphas_, self.noise_, self.lin_ = [], [], [], [], []
        for j in range(Y.shape[1]):
            ok = np.isfinite(Y[:, j])                     # NaN = below noise floor, ignored
            Xj, yj = Xs[ok], Y[ok, j]
            # explicit linear mean (log-current ~ linear in Vgs/Vt etc.), GP on the residual
            lin = Ridge(alpha=1e-6).fit(Xj, yj) if self.linear_mean else None
            self.lin_.append(lin)
            if lin is not None:
                yj = yj - lin.predict(Xj)
            idx = rng.choice(len(Xj), min(self.n_sub, len(Xj)), replace=False)
            kern = (ConstantKernel(1.0, (1e-2, 1e3)) * RBF(np.ones(X.shape[1]), (1e-2, 1e4))
                    + WhiteKernel(1e-6, (1e-10, 1e-2)))
            hp = GaussianProcessRegressor(kern, random_state=self.seed).fit(Xj[idx], yj[idx])
            full = GaussianProcessRegressor(hp.kernel_, optimizer=None).fit(Xj, yj)
            self.Xs_.append(Xj)
            self.kernels_.append(full.kernel_.k1)          # signal kernel (no white noise)
            self.alphas_.append(full.alpha_.ravel())
            self.noise_.append(full.kernel_.k2.noise_level)
        return self

    def predict(self, X):
        Xs = self.sx.transform(X)
        out = np.empty((len(Xs), len(self.alphas_)))
        for a in range(0, len(Xs), 2000):
            xb = Xs[a:a + 2000]
            for j, (k, al, Xt, lin) in enumerate(zip(self.kernels_, self.alphas_, self.Xs_, self.lin_)):
                out[a:a + 2000, j] = k(xb, Xt) @ al + (lin.predict(xb) if lin is not None else 0.0)
        return out


def make_models(seed=C.SEED_MODEL):
    from lightgbm import LGBMRegressor
    from sklearn.multioutput import MultiOutputRegressor
    return {
        "PolyRidge": make_pipeline(StandardScaler(), PolynomialFeatures(3), Ridge(alpha=1e-4)),
        "LightGBM": MultiOutputRegressor(LGBMRegressor(
            n_estimators=5000, learning_rate=0.02, num_leaves=15, min_child_samples=5,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1e-3,
            random_state=seed, verbose=-1, n_jobs=4)),
        "MLP": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 64, 64), activation="tanh", solver="adam", learning_rate_init=3e-3,
            batch_size=64, max_iter=1500, n_iter_no_change=60, tol=1e-9, alpha=1e-6,
            early_stopping=True, validation_fraction=0.1, random_state=seed)),
        "GP": GPLite(seed=seed),
    }


def channel_strength(df, case):
    """0 .. 1: how far the OFF device's source sits from the 'full sub-threshold' rail,
    as a fraction of VDD. NMOS OFF: source = low terminal, x = Vlo / VDD.
    PMOS OFF: source = high terminal, x = (VDD - Vhi) / VDD. Small x = channel-dominated."""
    vdd = df["vdd"].to_numpy(float)
    if case == "nmos_off":
        return np.minimum(df["vd"], df["vs"]).to_numpy(float) / vdd
    return 1.0 - np.maximum(df["vd"], df["vs"]).to_numpy(float) / vdd


class TransistorModel:
    """features -> regressor -> ln(|I|/W) of fixed-sign currents; mixed-sign one via KCL.

    OFF cases use two experts (same model type): one for the channel (sub-threshold)
    dominated regime and one for the regime where the "floor" currents (gate
    tunnelling + junction / body) dominate. In log-current the change-over is a sharp
    bend that one smooth model fits poorly. The split point x* in channel strength is
    found from the training data: the first x-bin where the median ratio
    |channel-carrying current| / (|Ig| + |Ib|) falls below 10. Each expert also
    trains on an OVERLAP band past the split.
    """
    OVERLAP, RATIO, BIN = 0.05, 10.0, 0.025

    def _find_split(self, df, F):
        main = 0 if self.case == "nmos_off" else 2         # channel current: drain side of the OFF device
        ratio = np.abs(F[:, main]) / (np.abs(F[:, 1]) + np.abs(F[:, 3]) + 1e-30)
        x = channel_strength(df, self.case)
        for lo in np.arange(0, 1, self.BIN):
            s = (x >= lo) & (x < lo + self.BIN)
            if s.sum() > 20 and np.median(ratio[s]) < self.RATIO:
                return float(lo)
        return 0.5

    def __init__(self, reg, case):
        self.reg, self.case = reg, case
        self.split = case.endswith("_off")

    def fit(self, df):
        from copy import deepcopy
        F = to_frame(df)
        pos = (F > 0).mean(axis=0)
        mixed = np.minimum(pos, 1 - pos)                  # 0 = always same sign
        # derive at most one current from KCL: the one whose sign varies most
        self.kcl_idx = int(np.argmax(mixed)) if mixed.max() > 0.005 else None
        self.learn = [k for k in range(4) if k != self.kcl_idx]
        self.sign = np.where(pos >= 0.5, 1.0, -1.0)
        Ik = np.abs(F[:, self.learn])
        Y = np.log(np.maximum(Ik, NOISE_FLOOR) / df["w"].to_numpy(float)[:, None])
        if isinstance(self.reg, GPLite):
            Y[Ik < NOISE_FLOOR] = np.nan                  # GP: drop noise-level samples
        X = features(df)[PHYS].to_numpy()
        if self.split:
            self.x_split = self._find_split(df, F)
            x = channel_strength(df, self.case)
            masks = [x < self.x_split + self.OVERLAP, x >= self.x_split - self.OVERLAP]
        else:
            masks = [np.ones(len(df), bool)]
        self.experts = []
        for mk in masks:
            mu, sd = np.nanmean(Y[mk], 0), np.nanstd(Y[mk], 0)
            reg = deepcopy(self.reg).fit(X[mk], (Y[mk] - mu) / sd)
            self.experts.append((reg, mu, sd))
        return self

    def predict_frame(self, df):
        X = features(df)[PHYS].to_numpy()
        if self.split:
            region = (channel_strength(df, self.case) >= self.x_split).astype(int)
        else:
            region = np.zeros(len(df), int)
        Y = np.zeros((len(df), len(self.learn)))
        for r, (reg, mu, sd) in enumerate(self.experts):
            sel = region == r
            if sel.any():
                Y[sel] = np.asarray(reg.predict(X[sel])).reshape(sel.sum(), -1) * sd + mu
        F = np.zeros((len(df), 4))
        F[:, self.learn] = self.sign[self.learn] * np.exp(Y) * df["w"].to_numpy(float)[:, None]
        if self.kcl_idx is not None:
            F[:, self.kcl_idx] = -F[:, self.learn].sum(axis=1)
        return F

    def predict(self, df):
        return from_frame(self.predict_frame(df), df)
