"""Experiment (exp_extra): do more training rows (our own extra data) improve accuracy?

Candidates per transistor case; all use the official features, hi/lo frame,
KCL-derived mixed-sign current and two-expert split (ml_utils.TransistorModel):
  GP_7k    the official models (models_trained/), 7,000 training rows
  GP_cap   GP experts trained on up to GP_MAX rows sampled from the 47k pool
  MLP_ens  average of 3 MLPs (3 x 128, tanh, Adam) trained on the whole pool
Pool = original training split (7,000) + exp_extra/data/<case>_extra_*.csv.
The best candidate per case is chosen on the ORIGINAL validation split; the
test split is scored once. Gates and the parity checker are evaluated on the
same 200 corners and SPICE references as the official results.
Writes only into exp_extra/out/. Resumable: fitted models are cached.
"""
import glob
import sys
import time
import warnings
from importlib import import_module
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import circuits as G  # noqa: E402
import config as C  # noqa: E402
import ml_utils as M  # noqa: E402

warnings.filterwarnings("ignore")
T = import_module("03_train_models")
summary = import_module("06_gate_ml").summary
OUT = HERE / "out"
(OUT / "models").mkdir(parents=True, exist_ok=True)
GP_MAX = 9000


class GPCap(M.GPLite):
    """GPLite on at most max_n randomly chosen rows (exact GP cost grows as n^3)."""

    def __init__(self, max_n=GP_MAX, seed=C.SEED_MODEL):
        super().__init__(seed=seed)
        self.max_n = max_n

    def fit(self, X, Y):
        Y = np.asarray(Y)
        if len(X) > self.max_n:
            idx = np.random.default_rng(self.seed).choice(len(X), self.max_n, replace=False)
            X, Y = X[idx], Y[idx]
        return super().fit(X, Y)


class MLPEns:
    """Average of n MLPs trained with different seeds."""

    def __init__(self, n=3, seed=C.SEED_MODEL):
        self.n, self.seed = n, seed

    def fit(self, X, Y):
        Y = np.asarray(Y).reshape(len(X), -1)
        self.m = [make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(128, 128, 128), activation="tanh", solver="adam",
            learning_rate_init=1e-3, batch_size=256, max_iter=600, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=30, tol=1e-7, alpha=1e-6,
            random_state=self.seed + i)).fit(X, Y) for i in range(self.n)]
        return self

    def predict(self, X):
        return np.mean([np.asarray(m.predict(X)).reshape(len(X), -1) for m in self.m], axis=0)


CANDS = {"GP_cap": GPCap, "MLP_ens": MLPEns}


def main():
    rows, chosen, allm = [], {}, {}
    for case in C.CASES:
        df = pd.read_csv(C.DATA_DIR / f"{case}.csv")
        tr, va, te = T.split(df)
        extra = sorted(glob.glob(str(HERE / "data" / f"{case}_extra_*.csv")))
        pool = pd.concat([tr] + [pd.read_csv(f) for f in extra], ignore_index=True)
        models = {"GP_7k": joblib.load(C.MODEL_DIR / f"{case}.joblib")}
        for name, cls in CANDS.items():
            f = OUT / "models" / f"{name}_{case}.joblib"
            if f.exists():
                models[name] = joblib.load(f)
                print(f"loaded {f.name}", flush=True)
                continue
            t0 = time.time()
            m = M.TransistorModel(cls(), case).fit(pool)
            joblib.dump(m, f)
            models[name] = m
            print(f"{case} {name}: trained on {len(pool)} rows in {time.time() - t0:.0f} s", flush=True)
        val = {}
        for name, m in models.items():
            r = T.metrics(m.predict(va), va[C.CURRENTS].to_numpy())
            r.update(case=case, model=name, split="val", train_rows=7000 if name == "GP_7k" else len(pool))
            rows.append(r)
            val[name] = r["mean_med%"]
            print(f"  {case:9s} {name:8s} val: mean-median {r['mean_med%']:.3f}%  "
                  + " ".join(f"{c}:{r[f'{c}_med%']:.2f}/{r[f'{c}_p95%']:.1f}" for c in C.CURRENTS), flush=True)
        best = min(val, key=val.get)
        rt = T.metrics(models[best].predict(te), te[C.CURRENTS].to_numpy())
        rt.update(case=case, model=best, split="test")
        rows.append(rt)
        chosen[case], allm[case] = best, models
        print(f"  -> {case}: chosen {best}, test mean-median {rt['mean_med%']:.3f}%", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "transistor_metrics.csv", index=False, float_format="%.4g")

    corners = G.load_corners().set_index("corner", drop=False)
    sp = pd.read_csv(C.RESULTS_DIR / "gate_spice.csv", dtype={"pattern": str})
    csp = pd.read_csv(C.RESULTS_DIR / "circuit_spice.csv", dtype={"vector": str})
    vectors = ["".join(v) for v in product("01", repeat=8)]
    cnt = pd.DataFrame({v: pd.Series([p for _, p in G.parity_logic(v)[1]]).value_counts()
                        .reindex(["00", "01", "10", "11"], fill_value=0) for v in vectors}).T
    sets = {"official_GP_7k": {c: "GP_7k" for c in C.CASES},
            "all_GP_cap": {c: "GP_cap" for c in C.CASES},
            "all_MLP_ens": {c: "MLP_ens" for c in C.CASES},
            "selected_on_val": chosen}
    res = []
    for sname, sel in sets.items():
        models = {c: allm[c][sel[c]] for c in C.CASES}
        parts = []
        for (gate, pat), d in sp.groupby(["gate", "pattern"], sort=False):
            d = d.sort_values("corner").copy()
            cs = corners.loc[d.corner.to_numpy()]
            node_v = {n: d[f"v_{n}"].to_numpy() for n in G.GATES[gate]["nodes"]}
            d["leak_ml"] = G.ml_gate_leakage(gate, pat, cs, node_v, models)
            parts.append(d)
        g = pd.concat(parts, ignore_index=True)
        g.to_csv(OUT / f"gate_ml_{sname}.csv", index=False, float_format="%.6e")
        summary(g, ["gate", "pattern"]).to_csv(OUT / f"gate_results_{sname}.csv", index=False, float_format="%.4g")
        ga = summary(g.assign(all="all"), ["all"]).iloc[0]
        nand = g[g.gate == "NAND2"].pivot(index="corner", columns="pattern", values="leak_ml")[["00", "01", "10", "11"]]
        ml = pd.DataFrame(nand.to_numpy() @ cnt.to_numpy().T, index=nand.index, columns=cnt.index)
        ml = ml.stack().rename("leak_ml").reset_index()
        ml.columns = ["corner", "vector", "leak_ml"]
        p = csp.merge(ml, on=["corner", "vector"])
        pa = summary(p.assign(all="all"), ["all"]).iloc[0]
        res.append(dict(set=sname, models=" ".join(f"{c}={sel[c]}" for c in C.CASES),
                        gate_median=ga.median_err_pct, gate_p95=ga.p95_err_pct, gate_bias=ga.bias_pct,
                        gate_accuracy=ga.accuracy_pct, parity_median=pa.median_err_pct, parity_p95=pa.p95_err_pct,
                        parity_bias=pa.bias_pct, parity_accuracy=pa.accuracy_pct))
        print(f"{sname:16s} gates: acc {ga.accuracy_pct:.2f}% med {ga.median_err_pct:.2f}% p95 {ga.p95_err_pct:.2f}% "
              f"bias {ga.bias_pct:+.2f}% | parity: acc {pa.accuracy_pct:.2f}% med {pa.median_err_pct:.2f}% "
              f"p95 {pa.p95_err_pct:.2f}% bias {pa.bias_pct:+.2f}%", flush=True)
    pd.DataFrame(res).to_csv(OUT / "summary.csv", index=False, float_format="%.4g")
    print("EXPERIMENT DONE", flush=True)


if __name__ == "__main__":
    main()
