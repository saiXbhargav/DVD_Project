"""Per-current validation metrics (median, p95, R2 on log) for our models, median, worst p95, log-R2, sign."""
import joblib
import numpy as np
import pandas as pd

from common import OUT, C
from common2 import T

rows = []
for case in C.CASES:
    _, va, te = T.split(pd.read_csv(C.DATA_DIR / f"{case}.csv"))
    names = [("official GP 7k", C.MODEL_DIR / f"{case}.joblib"), ("GPBag87", OUT / "models" / f"GPBag87_{case}.joblib")]
    if case.endswith("_on"):
        names.append(("GPBag87X", OUT / "models" / f"GPBag87X_{case}.joblib"))
    for name, f in names:
        m = joblib.load(f)
        P, Y = m.predict(va), va[C.CURRENTS].to_numpy()
        e = 100 * np.abs(P - Y) / np.abs(Y)
        sign = np.mean(np.sign(P) == np.sign(Y), axis=0) * 100
        lp, ly = np.log10(np.abs(P) + 1e-30), np.log10(np.abs(Y) + 1e-30)
        r2 = 1 - ((lp - ly) ** 2).sum(0) / ((ly - ly.mean(0)) ** 2).sum(0)
        big = np.abs(Y) >= 1e-11
        med_big = [np.median(e[big[:, j], j]) if big[:, j].any() else np.nan for j in range(4)]
        rows.append(dict(case=case, model=name,
                         median=" / ".join(f"{np.median(e[:, j]):.2f}" for j in range(4)),
                         worst_p95=f"{np.quantile(e, .95, axis=0).max():.1f} ({C.CURRENTS[np.argmax(np.quantile(e, .95, axis=0))]})",
                         r2_min=f"{r2.min():.4f}", sign_min=f"{sign.min():.1f}",
                         median_ge_1e11=" / ".join("-" if np.isnan(x) else f"{x:.2f}" for x in med_big)))
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))
