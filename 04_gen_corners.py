"""Step 4 (Parts B / C): the 200 PVT corners and their transistor widths.

  data/corners.csv    200 corners (seed 1234): VDD, T, one NMOS and one PMOS process-parameter
                      set, from the Part A distributions
  data/corners_w.csv  the same corners + one NMOS width wn and one PMOS width wp per corner
                      (seed 2468, uniform 16-128 nm = the project's width range); every NMOS of a
                      gate / circuit gets wn, every PMOS wp. This is the main (varying-width) setup.
Existing files are kept (skip-if-exists).
"""
import numpy as np
import pandas as pd

from final_method import CORNERS_W, DATA, SEED_W, C, S


def main():
    f = DATA / "corners.csv"
    if f.exists():
        print("exists: data/corners.csv", flush=True)
    else:
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
        df.to_csv(f, index=False, float_format="%.10e")
        print(f"saved data/corners.csv ({n} corners)", flush=True)

    if CORNERS_W.exists():
        print("exists: data/corners_w.csv", flush=True)
    else:
        base = pd.read_csv(f)
        rng = np.random.default_rng(SEED_W)
        base["wn"] = rng.uniform(*C.W_RANGE, len(base))
        base["wp"] = rng.uniform(*C.W_RANGE, len(base))
        base.to_csv(CORNERS_W, index=False, float_format="%.10e")
        print("saved data/corners_w.csv", flush=True)
    cor = pd.read_csv(CORNERS_W)
    print(f"{len(cor)} corners; VDD {cor.vdd.min():.3f}-{cor.vdd.max():.3f} V, T {cor.temp.min():.0f}-"
          f"{cor.temp.max():.0f} C, Wn {cor.wn.min()*1e9:.1f}-{cor.wn.max()*1e9:.1f} nm, "
          f"Wp {cor.wp.min()*1e9:.1f}-{cor.wp.max()*1e9:.1f} nm", flush=True)
    print("STEP 4 DONE", flush=True)


if __name__ == "__main__":
    main()
