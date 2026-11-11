"""All accuracies of the GPX setup (read-only; uses saved result files + one transistor pass)."""
import joblib
import numpy as np
import pandas as pd

from common import OUT, HERE, C
from common2 import T

ON = ("nmos_on", "pmos_on")
pd.set_option("display.width", 250)
print("=== TRANSISTORS (GPX: GPBag87X for ON, GPBag87 for OFF)")
rows = []
for case in C.CASES:
    m = joblib.load(OUT / "models" / f"{'GPBag87X' if case in ON else 'GPBag87'}_{case}.joblib")
    _, va, te = T.split(pd.read_csv(C.DATA_DIR / f"{case}.csv"))
    fr = pd.read_csv(HERE / "fresh" / f"{case}.csv")
    for split, d in (("val", va), ("test", te), ("fresh", fr)):
        P, Y = m.predict(d), d[C.CURRENTS].to_numpy()
        e = 100 * np.abs(P - Y) / np.abs(Y)
        big = np.abs(Y) >= 1e-11
        lp, ly = np.log10(np.abs(P) + 1e-30), np.log10(np.abs(Y) + 1e-30)
        ok = np.abs(Y) > 1e-16
        r2 = [1 - ((lp[ok[:, j], j] - ly[ok[:, j], j]) ** 2).sum() / ((ly[ok[:, j], j] - ly[ok[:, j], j].mean()) ** 2).sum()
              for j in range(4)]
        rows.append(dict(case=case, split=split,
                         median_id_ig_is_ib=" / ".join(f"{np.median(e[:, j]):.2f}" for j in range(4)),
                         p95_id_ig_is_ib=" / ".join(f"{np.quantile(e[np.isfinite(e[:, j]), j], .95):.1f}" for j in range(4)),
                         mean_of_medians=np.mean([np.median(e[:, j]) for j in range(4)]),
                         ge1e11_median=np.median(e[big]), ge1e11_p95=np.quantile(e[big], .95),
                         r2_log_min=min(r2)))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== GATES per case (official corners | fresh corners)")
a = pd.read_csv(OUT / "round2_gate_results_GPX.csv", dtype={"pattern": str})
f = pd.read_csv(HERE / "fresh" / "gate_results_GPX.csv", dtype={"pattern": str})
t = a[["gate", "pattern", "bias_pct_test", "median_err_pct_test",
       "p95_err_pct_test", "accuracy_pct_test"]].merge(
    f[["gate", "pattern", "accuracy_pct", "median_err_pct"]].rename(
        columns={"accuracy_pct": "fresh_acc", "median_err_pct": "fresh_med"}), on=["gate", "pattern"])
print(t.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
print("\n=== OVERALL")
print(pd.read_csv(OUT / "round2_summary.csv").query("set == 'GPX'").T.to_string())
print(pd.read_csv(HERE / "fresh" / "fresh_summary.csv").query("set == 'GPX'").T.to_string())
