import pandas as pd

opts = ["GP_7k_rule", "GP_cap_rule", "GPBag47_rule", "GPBag87_rule"]
t = None
for o in opts:
    d = pd.read_csv(f"out/gate_results_{o}.csv", dtype={"pattern": str})
    d = d[["gate", "pattern", "accuracy_pct_test", "median_err_pct_test", "accuracy_pct_val"]].rename(
        columns={"accuracy_pct_test": f"{o[:-5]}_acc", "median_err_pct_test": f"{o[:-5]}_med",
                 "accuracy_pct_val": f"{o[:-5]}_vacc"})
    t = d if t is None else t.merge(d, on=["gate", "pattern"])
pd.set_option("display.width", 250)
print(t.round(2).to_string(index=False))
