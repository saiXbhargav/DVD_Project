"""Part B/C step 0: draw the 200 PVT corners shared by all gates and the test circuit.

A corner is one draw of VDD, temperature, one NMOS process-parameter set and one
PMOS process-parameter set (spec section 5), using the section 4 distributions.
"""
import numpy as np
import pandas as pd

import config as C
import spice_utils as S


def main():
    rng = np.random.default_rng(C.SEED_CORNERS)
    n = C.N_CORNERS
    df = pd.DataFrame({
        "corner": np.arange(n),
        "vdd": rng.uniform((1 - C.VDD_SPREAD) * C.VDD_NOM, (1 + C.VDD_SPREAD) * C.VDD_NOM, n),
        "temp": rng.uniform(*C.T_RANGE, n),
    })
    for dev in ("n", "p"):
        for p in C.PROC_PARAMS:
            df[f"{dev}_{p}"] = S.NOMINAL[dev][p] * rng.uniform(1 - C.PROC_SPREAD, 1 + C.PROC_SPREAD, n)
    df.to_csv(C.DATA_DIR / "corners.csv", index=False, float_format="%.10e")
    print(f"saved data/corners.csv ({n} corners)")
    print(df.describe().T[["min", "max"]].to_string())


if __name__ == "__main__":
    main()
