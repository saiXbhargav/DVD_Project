"""Step 2 (Part A data check): Kirchhoff (KCL) check per case and three trend plots.

  results/kcl_check.csv              sum of the four terminal currents, the 10,000-row datasets
  results/kcl_check_train_extra.csv  the same for the 80,000 extra training rows per case
  figures/trend_ioff_vs_temp.png     OFF current vs temperature
  figures/trend_ioff_vs_vth0.png     OFF current vs vth0 (T = 27 and 125 C)
  figures/trend_ig_vs_toxe.png       gate current vs toxe (ON devices)
Each trend plot shows the random dataset (left) and a controlled SPICE sweep at nominal process
and VDD (right), so the physical trend is visible:
  OFF bias: NMOS Vd = VDD, Vs = 0; PMOS Vs = VDD, Vd = 0.  ON bias for Ig: NMOS Vd = Vs = 0,
  PMOS Vd = Vs = VDD.  Sweep width W = 4 Lmin.
"""
import glob
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from final_method import EXTRA, C, S  # noqa: E402

gen = importlib.import_module("01_gen_transistor_data")
W_SWEEP = 4 * C.LMIN
COLORS = {"nmos_off": "#2a78d6", "pmos_off": "#eb6834", "nmos_on": "#2a78d6", "pmos_on": "#eb6834"}


def load(case):
    return pd.read_csv(C.DATA_DIR / f"{case}.csv")


def kcl_rows(case, d):
    I = d[C.CURRENTS]
    res = I.sum(axis=1).abs()
    rel = res / I.abs().max(axis=1)
    return {"case": case, "n": len(d),
            "max_abs_residual_A": res.max(), "median_abs_residual_A": res.median(),
            "max_rel_residual": rel.max(), "median_rel_residual": rel.median(),
            "frac_rel_below_1e-6": (rel < 1e-6).mean(),
            "frac_abs_below_1e-15A": (res < 1e-15).mean()}


def kcl_tables():
    t = pd.DataFrame([kcl_rows(case, load(case)) for case in C.CASES])
    t.to_csv(C.RESULTS_DIR / "kcl_check.csv", index=False)
    x = pd.DataFrame([kcl_rows(case, pd.concat([pd.read_csv(f) for f in
                                                sorted(glob.glob(str(EXTRA / f"{case}_*.csv")))]))
                      for case in C.CASES])
    x.to_csv(C.RESULTS_DIR / "kcl_check_train_extra.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3e}".format):
        print("datasets (10,000 rows per case)")
        print(t.to_string(index=False))
        print("extra training rows (80,000 per case)")
        print(x.to_string(index=False))


def sweep(case, temp=27.0, param=None, values=None, vd=None, vs=None):
    """Controlled bias points at nominal parameters, simulated with SPICE."""
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


def style(ax, xlabel, ylabel):
    for a in ax:
        a.set_yscale("log")
        a.set_xlabel(xlabel)
        a.set_ylabel(ylabel)
        a.grid(alpha=.3, which="both")
        a.legend(fontsize=8, frameon=False)


def trend_plots():
    vdd = C.VDD_NOM
    off_bias = {"nmos_off": dict(vd=vdd, vs=0.0), "pmos_off": dict(vd=0.0, vs=vdd)}
    on_bias = {"nmos_on": dict(vd=0.0, vs=0.0), "pmos_on": dict(vd=vdd, vs=vdd)}

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for case, b in off_bias.items():
        d = load(case)
        big = (d.vd - d.vs).abs() > 0.8 * d.vdd
        ax[0].scatter(d.temp[big], d.id[big].abs(), s=3, alpha=0.25, color=COLORS[case],
                      label=f"{case} dataset (|Vds| > 0.8 VDD)")
        s = sweep(case, param="temp", values=np.linspace(*C.T_RANGE, 34), **b)
        ax[1].plot(s.temp, s.id.abs(), "-o", ms=3, color=COLORS[case], label=f"{case}, nominal, W = 4L, |Vds| = VDD")
    style(ax, "Temperature (°C)", "|I_off| = |Id| (A)")
    ax[0].set_title("Dataset")
    ax[1].set_title("Controlled SPICE sweep")
    fig.suptitle("OFF current vs temperature (16nm HP)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "trend_ioff_vs_temp.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for case, b in off_bias.items():
        dev = C.CASES[case][0]
        d = load(case)
        big = (d.vd - d.vs).abs() > 0.8 * d.vdd
        ax[0].scatter(d.vth0[big].abs(), d.id[big].abs(), s=3, alpha=0.25, color=COLORS[case], label=f"{case} dataset")
        vals = S.NOMINAL[dev]["vth0"] * np.linspace(1 - C.PROC_SPREAD, 1 + C.PROC_SPREAD, 21)
        for T, ls in ((27.0, "-"), (125.0, "--")):
            s = sweep(case, param="vth0", values=vals, temp=T, **b)
            ax[1].plot(np.abs(s.vth0), s.id.abs(), ls, marker="o", ms=3, color=COLORS[case], label=f"{case}, T = {T:.0f} °C")
    style(ax, "|vth0| (V)", "|I_off| = |Id| (A)")
    ax[0].set_title("Dataset (|Vds| > 0.8 VDD)")
    ax[1].set_title("Controlled SPICE sweep, nominal, W = 4L, |Vds| = VDD")
    fig.suptitle("OFF current vs vth0 (16nm HP)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "trend_ioff_vs_vth0.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for case, b in on_bias.items():
        dev = C.CASES[case][0]
        d = load(case)
        ax[0].scatter(d.toxe * 1e9, d.ig.abs(), s=3, alpha=0.25, color=COLORS[case], label=f"{case} dataset")
        vals = S.NOMINAL[dev]["toxe"] * np.linspace(1 - C.PROC_SPREAD, 1 + C.PROC_SPREAD, 21)
        s = sweep(case, param="toxe", values=vals, **b)
        ax[1].plot(s.toxe * 1e9, s.ig.abs(), "-o", ms=3, color=COLORS[case], label=f"{case}, nominal, W = 4L, T = 27 °C")
    style(ax, "toxe (nm)", "|Ig| (A)")
    ax[0].set_title("Dataset (all biases)")
    ax[1].set_title("Controlled SPICE sweep, |Vgs| = |Vgd| = VDD")
    fig.suptitle("Gate current vs toxe (16nm HP)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "trend_ig_vs_toxe.png", dpi=150)
    plt.close(fig)
    print("saved figures/trend_ioff_vs_temp.png, trend_ioff_vs_vth0.png, trend_ig_vs_toxe.png", flush=True)


def main():
    kcl_tables()
    trend_plots()
    print("STEP 2 DONE", flush=True)


if __name__ == "__main__":
    main()
