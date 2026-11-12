"""Step 3 (Part A models): model comparison, then the final transistor models and their scores.

a) Model comparison (deliverable 4): the four model types of ml_utils.make_models (polynomial
   ridge, LightGBM, MLP, exact GP), each wrapped in ml_utils.TransistorModel, trained on the
   7,000-row training split of each case and scored on the validation split (R2 on log|I|, median
   and 95th-percentile relative error per current) -> results/model_comparison_val.csv.
   The GP is the best type in all four cases. Kept if it exists (~1 h to regenerate).
b) Final models: the chosen type, improved on the validation split (history: old/exp_extra,
   old/exp_acc): averaged GP on the 87,000-row pool (training split + data/train_extra), plain
   Vgs, Vgd, Vds as extra features for the ON cases -> models_trained/<case>.joblib.
   Kept if it exists (skip-if-exists).
c) Final models scored on the original validation and test splits -> results/transistor_metrics.csv.
"""
import os
from pathlib import Path

if all((Path(__file__).resolve().parent / "models_trained" / f"{c}.joblib").exists()
       for c in ("nmos_on", "nmos_off", "pmos_on", "pmos_off")):
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")       # scoring only: one thread = bit-reproducible numbers

import time  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from final_method import (MODELS, RES, C, M, load_models, new_model, split, training_pool,  # noqa: E402
                          transistor_metrics)


def comparison_metrics(P, T):
    from sklearn.metrics import r2_score
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


def model_comparison():
    f = RES / "model_comparison_val.csv"
    if f.exists():
        print("exists: results/model_comparison_val.csv", flush=True)
        return
    warnings.filterwarnings("ignore")
    comp = []
    for case in C.CASES:
        tr, va, _ = split(pd.read_csv(C.DATA_DIR / f"{case}.csv"))
        for name, reg in M.make_models().items():
            t0 = time.time()
            m = M.TransistorModel(reg, case).fit(tr)
            r = comparison_metrics(m.predict(va), va[C.CURRENTS].to_numpy())
            r.update(case=case, model=name, train_s=round(time.time() - t0, 1),
                     kcl_derived=M.FRAME[m.kcl_idx] if m.kcl_idx is not None else "-")
            comp.append(r)
            print(f"{case:9s} {name:10s} validation mean of medians {r['mean_med%']:6.2f}% ({r['train_s']} s)",
                  flush=True)
    cols = (["case", "model"] + [f"{c}_{k}" for c in C.CURRENTS for k in ("r2", "med%", "p95%")]
            + ["mean_med%", "kcl_derived", "train_s"])
    pd.DataFrame(comp)[cols].to_csv(f, index=False, float_format="%.4g")
    print("saved results/model_comparison_val.csv", flush=True)


def main():
    model_comparison()
    for case in C.CASES:
        f = MODELS / f"{case}.joblib"
        if f.exists():
            print(f"exists: models_trained/{f.name}", flush=True)
            continue
        t0 = time.time()
        pool = training_pool(case)
        m = new_model(case).fit(pool)
        tmp = f.with_suffix(".tmp")
        joblib.dump(m, tmp)
        tmp.replace(f)
        print(f"{case}: trained on {len(pool)} rows, {time.time() - t0:.0f} s", flush=True)

    models, rows = load_models(), []
    for case in C.CASES:
        _, va, te = split(pd.read_csv(C.DATA_DIR / f"{case}.csv"))
        for name, d in (("validation", va), ("test", te)):
            r, _ = transistor_metrics(models[case], d)
            rows.append(dict(case=case, split=name, n=len(d), **r))
    t = pd.DataFrame(rows)
    t.to_csv(RES / "transistor_metrics.csv", index=False, float_format="%.6g")
    with pd.option_context("display.width", 250):
        print(t[["case", "split", "id_median_pct", "ig_median_pct", "is_median_pct", "ib_median_pct",
                 "mean_of_medians_pct", "ge1e11_median_pct"]].round(3).to_string(index=False))
    print("STEP 3 DONE", flush=True)


if __name__ == "__main__":
    main()
