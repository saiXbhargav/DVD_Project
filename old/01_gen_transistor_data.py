"""Part A, step 1: sample PVT + bias points and simulate single NMOS/PMOS transistors.

For each case (NMOS ON/OFF, PMOS ON/OFF) draw N samples (VDD first, then Vd, Vs,
W, T and the five process parameters, all independent and uniform), build a
single-FET netlist with one voltage source per terminal, run ngspice and store
the terminal currents.

Current sign convention: id, ig, is, ib are the raw ngspice source currents
I(Vd), I(Vg), I(Vs), I(Vb), i.e. positive = current flowing OUT of the device
terminal into the source (same convention as the TA's sample database).
KCL: id + ig + is + ib = 0.

Usage: python3 01_gen_transistor_data.py [N]
"""
import sys
import time

import numpy as np
import pandas as pd

import config as C
import spice_utils as S


def sample_case(case, n, rng):
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


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else C.N_SAMPLES
    print(f"simulator: {S.ngspice_version()}, model: {C.MODEL_FILE.name}, N = {n}")
    for k, case in enumerate(C.CASES):
        dev = C.CASES[case][0]
        rng = np.random.default_rng(C.SEED_DATA + k)
        df = sample_case(case, n, rng)
        nets = [netlist(r, dev, case) for r in df.to_dict("records")]
        (C.NETLIST_DIR / f"{case}_example.cir").write_text(nets[0])

        t0 = time.time()
        I = S.run_op(nets, ["i(vd)", "i(vg)", "i(vs)", "i(vb)"])
        for j, c in enumerate(C.CURRENTS):
            df[c] = I[:, j]
        bad = df[C.CURRENTS].isna().any(axis=1)
        if bad.any():
            print(f"  WARNING {case}: {bad.sum()} failed simulations dropped")
            df = df[~bad]
        df[C.CSV_COLUMNS].to_csv(C.DATA_DIR / f"{case}.csv", index=False, float_format="%.10e")
        print(f"{case:9s}: {len(df)} rows in {time.time() - t0:.1f} s -> data/{case}.csv")


if __name__ == "__main__":
    main()
