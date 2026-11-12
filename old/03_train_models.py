"""Part A step 2: train and compare 4 model types for each of the 4 transistor cases.

Split: 70 / 15 / 15 (train / validation / test) with a fixed seed.
Metrics per terminal current (Id, Ig, Is, Ib):
  * R2 on log10|I|
  * median and 95th-percentile relative error |I_pred - I_spice| / |I_spice|
The best model per case is chosen on validation (lowest mean of the four median
errors), refit is not needed (it was trained on the training split only), and its
test-set errors are reported. Saves the chosen models, the comparison table, the
test table and predicted-vs-SPICE plots.

Usage: python3 03_train_models.py [case,case,...] [model,model,...]
"""
import sys
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import config as C
import ml_utils as M

warnings.filterwarnings("ignore")


def split(df):
    tr, rest = train_test_split(df, test_size=1 - C.SPLIT[0], random_state=C.SEED_SPLIT)
    va, te = train_test_split(rest, test_size=C.SPLIT[2] / (C.SPLIT[1] + C.SPLIT[2]), random_state=C.SEED_SPLIT)
    return tr, va, te


def metrics(P, T):
    out = {}
    e = M.rel_err(P, T)
    for j, c in enumerate(C.CURRENTS):
        lt, lp = np.log10(np.abs(T[:, j]) + 1e-30), np.log10(np.abs(P[:, j]) + 1e-30)
        ok = np.isfinite(e[:, j])
        out[f"{c}_r2"] = r2_score(lt, lp)
        out[f"{c}_med%"] = 100 * np.median(e[ok, j])
        out[f"{c}_p95%"] = 100 * np.percentile(e[ok, j], 95)
    out["mean_med%"] = np.mean([out[f"{c}_med%"] for c in C.CURRENTS])
    return out


def parity_plot(case, P, T, name):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for j, (c, ax) in enumerate(zip(C.CURRENTS, axs)):
        t, p = np.abs(T[:, j]), np.abs(P[:, j])
        ax.scatter(t, p, s=3, alpha=0.4)
        lo, hi = max(min(t.min(), p.min()), 1e-18), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"SPICE |{c}| (A)"); ax.set_ylabel(f"ML |{c}| (A)")
        e = M.rel_err(P[:, j], T[:, j])
        ax.set_title(f"{c}: median {100*np.nanmedian(e):.2f}%, p95 {100*np.nanpercentile(e, 95):.1f}%", fontsize=9)
        ax.grid(alpha=.3, which="both")
    fig.suptitle(f"{case}: {name}, test set (n={len(T)})")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / f"pred_vs_spice_{case}.png", dpi=130)
    plt.close(fig)


def main():
    cases = sys.argv[1].split(",") if len(sys.argv) > 1 else list(C.CASES)
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    comp, test_rows = [], []
    for case in cases:
        df = pd.read_csv(C.DATA_DIR / f"{case}.csv")
        tr, va, te = split(df)
        fitted = {}
        for name, reg in M.make_models().items():
            if only and name not in only:
                continue
            t0 = time.time()
            m = M.TransistorModel(reg, case).fit(tr)
            r = metrics(m.predict(va), va[C.CURRENTS].to_numpy())
            r.update(case=case, model=name, train_s=round(time.time() - t0, 1),
                     kcl_derived=M.FRAME[m.kcl_idx] if m.kcl_idx is not None else "-")
            comp.append(r)
            fitted[name] = (m, r)
            print(f"{case:9s} {name:12s} val mean-median {r['mean_med%']:6.2f}%  "
                  + " ".join(f"{c}:{r[f'{c}_med%']:.2f}/{r[f'{c}_p95%']:.1f}" for c in C.CURRENTS)
                  + f"  ({r['train_s']}s)", flush=True)
        best = min(fitted, key=lambda k: fitted[k][1]["mean_med%"])
        m = fitted[best][0]
        P, T = m.predict(te), te[C.CURRENTS].to_numpy()
        rt = metrics(P, T)
        rt.update(case=case, model=best)
        test_rows.append(rt)
        joblib.dump(m, C.MODEL_DIR / f"{case}.joblib")
        parity_plot(case, P, T, best)
        print(f"  -> chosen {best}; test mean-median {rt['mean_med%']:.2f}%", flush=True)

    cols = ["case", "model"] + [f"{c}_{k}" for c in C.CURRENTS for k in ("r2", "med%", "p95%")] + ["mean_med%"]
    pd.DataFrame(comp)[cols + ["kcl_derived", "train_s"]].to_csv(C.RESULTS_DIR / "model_comparison_val.csv", index=False, float_format="%.4g")
    pd.DataFrame(test_rows)[cols].to_csv(C.RESULTS_DIR / "chosen_models_test.csv", index=False, float_format="%.4g")
    print("saved results/model_comparison_val.csv, results/chosen_models_test.csv, models_trained/*.joblib")


if __name__ == "__main__":
    main()
