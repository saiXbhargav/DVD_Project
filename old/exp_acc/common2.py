"""exp_acc round 2: shared pieces (new file; round-1 files are unchanged).

  FEATS_X   official physics features + plain (not kT/q-scaled) Vgs, Vgd, Vds. Gate
            tunnelling depends on the oxide voltage, not on kT/q; with only scaled
            features the ON-device gate current at Vds = 0 and cold corners lies
            outside the training feature range (NOR2 11 diagnosis).
  TMX       ml_utils.TransistorModel with a chosen feature list.
  MLPLin    linear (ridge) fit + one MLP per output on its residual (standard
            recipe: 3 x 128 tanh, 4000 epochs, patience 200); MLPEns = average of seeds.
  pools()   training pools (original training split + our extra data).
"""
import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from common import DATA, HERE, C, M  # noqa: F401
from importlib import import_module

T = import_module("03_train_models")
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


class MLPLin:
    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, Y):
        import time
        Y = np.asarray(Y).reshape(len(X), -1)
        self.sx = StandardScaler().fit(X)
        Xs = self.sx.transform(X)
        self.parts = []
        for j in range(Y.shape[1]):
            t0 = time.time()
            lin = Ridge(alpha=1e-3).fit(Xs, Y[:, j])
            mlp = MLPRegressor(hidden_layer_sizes=(128, 128, 128), activation="tanh", alpha=1e-6,
                               learning_rate_init=2e-3, max_iter=4000, tol=1e-10, n_iter_no_change=200,
                               early_stopping=True, validation_fraction=0.1, random_state=self.seed)
            mlp.fit(Xs, Y[:, j] - lin.predict(Xs))
            self.parts.append((lin, mlp))
            print(f"      output {j} seed {self.seed}: {len(X)} rows, {mlp.n_iter_} epochs, "
                  f"{time.time() - t0:.0f} s", flush=True)
        return self

    def predict(self, X):
        Xs = self.sx.transform(X)
        return np.column_stack([lin.predict(Xs) + mlp.predict(Xs) for lin, mlp in self.parts])


class MLPEns:
    def __init__(self, n=3):
        self.n = n

    def fit(self, X, Y):
        self.m = [MLPLin(seed=s).fit(X, Y) for s in range(self.n)]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.m], axis=0)


class Blend:
    """Average of the terminal currents predicted by several transistor models."""

    def __init__(self, members):
        self.members = members

    def predict(self, df):
        return np.mean([m.predict(df) for m in self.members], axis=0)


def pool87(case):
    tr, _, _ = T.split(pd.read_csv(C.DATA_DIR / f"{case}.csv"))
    files = sorted(glob.glob(str(HERE.parent / "exp_extra" / "data" / f"{case}_extra_*.csv"))) \
        + sorted(glob.glob(str(DATA / f"{case}_more_*.csv")))
    assert len(files) == 8, files
    return pd.concat([tr] + [pd.read_csv(f) for f in files], ignore_index=True)
