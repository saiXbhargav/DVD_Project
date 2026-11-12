"""Step 1 (Part A data): single-transistor SPICE data for the four cases.

  data/<case>.csv                      10,000 rows per case (seed 2026 + case index); split
                                       70 / 15 / 15 into train / validation / test in step 3
  data/train_extra/<case>_extra_0b.csv 4 x 10,000 more training rows (seed 5000 + 100 k + b)
  data/train_extra/<case>_more_0b.csv  4 x 10,000 more training rows (seed 7000 + 100 k + b)
Every row: VDD, T, W, Vd, Vs and the five process parameters drawn independently and uniformly
(VDD +-10 %, T -40..125 C, W 16-128 nm, Vd, Vs in [0, VDD], parameters +-10 %); Vg and Vb fixed by
the case (config.CASES). The extra rows are training rows only; validation and test rows are
always the original splits. Existing files are kept (skip-if-exists).
Current sign: id, ig, is, ib = ngspice source currents, > 0 = out of the device; KCL sum = 0.
"""
import time

import numpy as np

from final_method import BLOCK_ROWS, BLOCKS, DATA, EXTRA, SEED_EXTRA, SEED_MORE, C, S


def sample_case(case, n, rng):
    import pandas as pd
    dev, vg_rule, vb_rule = C.CASES[case]
    vdd = rng.uniform((1 - C.VDD_SPREAD) * C.VDD_NOM, (1 + C.VDD_SPREAD) * C.VDD_NOM, n)
    df = pd.DataFrame({
        "vdd": vdd,
        "temp": rng.uniform(*C.T_RANGE, n),
        "w": rng.uniform(*C.W_RANGE, n),
        "vd": rng.uniform(0, 1, n) * vdd,
        "vs": rng.uniform(0, 1, n) * vdd,
    })
    df["vg"] = vdd if vg_rule == "vdd" else vg_rule
    df["vb"] = vdd if vb_rule == "vdd" else vb_rule
    for p in C.PROC_PARAMS:
        df[p] = S.NOMINAL[dev][p] * rng.uniform(1 - C.PROC_SPREAD, 1 + C.PROC_SPREAD, n)
    return df


def netlist(row, dev, title=""):
    params = {p: row[p] for p in C.PROC_PARAMS}
    card = S.model_card(nparams=params) if dev == "n" else S.model_card(pparams=params)
    model = "nmos" if dev == "n" else "pmos"
    return "\n".join([
        f"* single {model.upper()} {title}",
        card,
        C.SPICE_OPTIONS,
        f".temp {S.fmt(row['temp'])}",
        f"M1 d g s b {model} W={S.fmt(row['w'])} L={S.fmt(C.LMIN)}",
        f"Vd d 0 {S.fmt(row['vd'])}",
        f"Vg g 0 {S.fmt(row['vg'])}",
        f"Vs s 0 {S.fmt(row['vs'])}",
        f"Vb b 0 {S.fmt(row['vb'])}",
        ".op",
        ".end",
    ]) + "\n"


def simulate(case, n, seed, f, example=False):
    if f.exists():
        print(f"exists: {f.relative_to(C.ROOT)}", flush=True)
        return
    dev = C.CASES[case][0]
    df = sample_case(case, n, np.random.default_rng(seed))
    nets = [netlist(r, dev, case) for r in df.to_dict("records")]
    if example:
        (C.NETLIST_DIR / f"{case}_example.cir").write_text(nets[0])
    t0 = time.time()
    I = S.run_op(nets, ["i(vd)", "i(vg)", "i(vs)", "i(vb)"])
    for j, c in enumerate(C.CURRENTS):
        df[c] = I[:, j]
    bad = df[C.CURRENTS].isna().any(axis=1)
    if bad.any():
        print(f"  WARNING {f.name}: {bad.sum()} failed simulations dropped", flush=True)
        df = df[~bad]
    tmp = f.with_suffix(".tmp")
    df[C.CSV_COLUMNS].to_csv(tmp, index=False, float_format="%.10e")
    tmp.replace(f)
    print(f"{f.relative_to(C.ROOT)}: {len(df)} rows, max KCL residual "
          f"{df[C.CURRENTS].sum(axis=1).abs().max():.1e} A, {time.time() - t0:.0f} s", flush=True)


def main():
    print(f"simulator: {S.ngspice_version()}, model: {C.MODEL_FILE.name}", flush=True)
    for k, case in enumerate(C.CASES):
        simulate(case, C.N_SAMPLES, C.SEED_DATA + k, DATA / f"{case}.csv", example=True)
        for b in range(1, BLOCKS + 1):
            simulate(case, BLOCK_ROWS, SEED_EXTRA + 100 * k + b, EXTRA / f"{case}_extra_{b:02d}.csv")
        for b in range(1, BLOCKS + 1):
            simulate(case, BLOCK_ROWS, SEED_MORE + 100 * k + b, EXTRA / f"{case}_more_{b:02d}.csv")
    print("STEP 1 DONE", flush=True)


if __name__ == "__main__":
    main()
