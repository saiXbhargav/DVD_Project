"""Part A data check: Kirchhoff (KCL) check per case and three trend plots.

Trend plots overlay the random dataset (grey points) with a controlled SPICE sweep
at nominal process / VDD (lines), so the physical trend is visible:
  1. OFF current vs temperature   (NMOS OFF: Vd=VDD, Vs=0; PMOS OFF: Vs=VDD, Vd=0)
  2. OFF current vs vth0          (same bias, T = 27 and 125 degC)
  3. Gate current vs toxe         (ON devices, Vd=Vs=0 for NMOS, Vd=Vs=VDD for PMOS)
"""
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import spice_utils as S

gen = importlib.import_module("01_gen_transistor_data")
W_SWEEP = 4 * C.LMIN


def load(case):
    return pd.read_csv(C.DATA_DIR / f"{case}.csv")


# ------------------------------------------------------------------ KCL
def kcl_table():
    rows = []
    for case in C.CASES:
        d = load(case)
        I = d[C.CURRENTS]
        res = I.sum(axis=1).abs()
        rel = res / I.abs().max(axis=1)
        rows.append({
            "case": case, "n": len(d),
            "max_abs_residual_A": res.max(), "median_abs_residual_A": res.median(),
            "max_rel_residual": rel.max(), "median_rel_residual": rel.median(),
            "frac_rel_below_1e-6": (rel < 1e-6).mean(),
            "frac_abs_below_1e-15A": (res < 1e-15).mean(),
        })
    t = pd.DataFrame(rows)
    t.to_csv(C.RESULTS_DIR / "kcl_check.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3e}".format):
        print(t.to_string(index=False))


# ------------------------------------------------------------------ controlled sweeps
def sweep(case, n=41, temp=27.0, fac=None, param=None, values=None, vd=None, vs=None):
    """Build a DataFrame of controlled bias points at nominal parameters and simulate it."""
    dev, vg_rule, vb_rule = C.CASES[case]
    vdd = C.VDD_NOM
    base = {"vdd": vdd, "temp": temp, "w": W_SWEEP, "vd": vd, "vs": vs,
            "vg": vdd if vg_rule == "vdd" else vg_rule,
            "vb": vdd if vb_rule == "vdd" else vb_rule}
    base.update(S.NOMINAL[dev])
    rows = []
    for v in values:
        r = dict(base)
        r[param] = v
        rows.append(r)
    df = pd.DataFrame(rows)
    nets = [gen.netlist(r, dev, case) for r in df.to_dict("records")]
    I = S.run_op(nets, ["i(vd)", "i(vg)", "i(vs)", "i(vb)"], n_jobs=1)
    for j, c in enumerate(C.CURRENTS):
        df[c] = I[:, j]
    return df


def trend_plots():
    vdd = C.VDD_NOM
    off_bias = {"nmos_off": dict(vd=vdd, vs=0.0), "pmos_off": dict(vd=0.0, vs=vdd)}
    on_bias = {"nmos_on": dict(vd=0.0, vs=0.0), "pmos_on": dict(vd=vdd, vs=vdd)}
    colors = {"nmos_off": "C0", "pmos_off": "C3", "nmos_on": "C0", "pmos_on": "C3"}

    # 1. OFF current vs temperature
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for case, b in off_bias.items():
        d = load(case)
        big = (d.vd - d.vs).abs() > 0.8 * d.vdd
        ax[0].scatter(d.temp[big], d.id[big].abs(), s=3, alpha=0.25, color=colors[case], label=f"{case} dataset (|Vds|>0.8VDD)")
        s = sweep(case, param="temp", values=np.linspace(*C.T_RANGE, 34), **b)
        ax[1].plot(s.temp, s.id.abs(), "-o", ms=3, color=colors[case], label=f"{case}, nominal, W=4L, |Vds|=VDD")
    for a in ax:
        a.set_yscale("log"); a.set_xlabel("Temperature (°C)"); a.set_ylabel("|I_off| = |Id| (A)"); a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
    ax[0].set_title("Dataset"); ax[1].set_title("Controlled SPICE sweep")
    fig.suptitle("OFF current vs temperature (16nm HP)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "trend_ioff_vs_temp.png", dpi=150); plt.close(fig)

    # 2. OFF current vs vth0
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for case, b in off_bias.items():
        dev = C.CASES[case][0]
        d = load(case)
        big = (d.vd - d.vs).abs() > 0.8 * d.vdd
        ax[0].scatter(d.vth0[big].abs(), d.id[big].abs(), s=3, alpha=0.25, color=colors[case], label=f"{case} dataset")
        nom = S.NOMINAL[dev]["vth0"]
        vals = nom * np.linspace(1 - C.PROC_SPREAD, 1 + C.PROC_SPREAD, 21)
        for T, ls in ((27.0, "-"), (125.0, "--")):
            s = sweep(case, param="vth0", values=vals, temp=T, **b)
            ax[1].plot(np.abs(s.vth0), s.id.abs(), ls, marker="o", ms=3, color=colors[case], label=f"{case}, T={T:.0f}°C")
    for a in ax:
        a.set_yscale("log"); a.set_xlabel("|vth0| (V)"); a.set_ylabel("|I_off| = |Id| (A)"); a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
    ax[0].set_title("Dataset (|Vds|>0.8VDD)"); ax[1].set_title("Controlled SPICE sweep, nominal, W=4L, |Vds|=VDD")
    fig.suptitle("OFF current vs vth0 (16nm HP)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "trend_ioff_vs_vth0.png", dpi=150); plt.close(fig)

    # 3. Gate current vs toxe
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for case, b in on_bias.items():
        dev = C.CASES[case][0]
        d = load(case)
        ax[0].scatter(d.toxe * 1e9, d.ig.abs(), s=3, alpha=0.25, color=colors[case], label=f"{case} dataset")
        nom = S.NOMINAL[dev]["toxe"]
        vals = nom * np.linspace(1 - C.PROC_SPREAD, 1 + C.PROC_SPREAD, 21)
        s = sweep(case, param="toxe", values=vals, **b)
        ax[1].plot(s.toxe * 1e9, s.ig.abs(), "-o", ms=3, color=colors[case], label=f"{case}, nominal, W=4L, T=27°C")
    for a in ax:
        a.set_yscale("log"); a.set_xlabel("toxe (nm)"); a.set_ylabel("|Ig| (A)"); a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
    ax[0].set_title("Dataset (all biases)"); ax[1].set_title("Controlled SPICE sweep, |Vgs|=|Vgd|=VDD")
    fig.suptitle("Gate current vs toxe (16nm HP)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "trend_ig_vs_toxe.png", dpi=150); plt.close(fig)
    print("saved figures/trend_ioff_vs_temp.png, trend_ioff_vs_vth0.png, trend_ig_vs_toxe.png")


if __name__ == "__main__":
    kcl_table()
    trend_plots()
