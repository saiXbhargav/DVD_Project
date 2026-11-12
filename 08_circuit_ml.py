"""Step 8 (Part C): ML leakage of the 8-bit parity checker and the overall summary.

Prescribed method: for every input vector, logic simulation gives the input pattern of each of the
28 NAND2; the circuit leakage is the sum of their NAND2 leakages (from step 6, same corner).
Done for version A and version B (main setup) and version A (fixed-size reference), and compared
with the parity checker's own SPICE simulation (step 7).
Outputs (results/):
  circuit_ml.csv       per corner and vector: SPICE leakage, leak_A, leak_B
  circuit_results.csv  test-circuit table (deliverable 8): SPICE median, ML median, bias, median and
                       95th-percentile error, accuracy, per setup / version
  parity_by_ones.csv   accuracy by the number of 1s in the input vector
  summary.csv          gates and parity checker: accuracy, median, p95, bias for every setup / version
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pandas as pd  # noqa: E402

from final_method import RES, parity_sum, summary, vec_counts  # noqa: E402

VERSIONS = {"A": "A: SPICE node voltages", "B": "B: ML node voltages"}


def row(setup, version, ag, ap):
    return dict(setup=setup, version=version, gate_acc=ag.accuracy_pct, gate_median=ag.median_err_pct,
                gate_p95=ag.p95_err_pct, gate_bias=ag.bias_pct, parity_acc=ap.accuracy_pct,
                parity_median=ap.median_err_pct, parity_p95=ap.p95_err_pct, parity_bias=ap.bias_pct)


def main():
    cnt, rows = vec_counts(), []
    g = pd.read_csv(RES / "gate_ml.csv", dtype={"pattern": str})
    psp = pd.read_csv(RES / "circuit_spice_w.csv", dtype={"vector": str})
    psp = psp[psp.corner.isin(g.corner)]
    circ = None
    for ver in ("A", "B"):
        ag = summary(g.rename(columns={f"leak_{ver}": "leak_ml"}).assign(all=1), ["all"]).iloc[0]
        nand = g[g.gate == "NAND2"].pivot(index="corner", columns="pattern", values=f"leak_{ver}")
        p = parity_sum(nand, cnt, psp)
        ap = summary(p.assign(all=1), ["all"]).iloc[0]
        rows.append(row("varying widths", VERSIONS[ver], ag, ap))
        p = p[["corner", "vector", "leak_spice", "leak_ml"]].rename(columns={"leak_ml": f"leak_{ver}"})
        circ = p if circ is None else circ.merge(p, on=["corner", "vector", "leak_spice"])
        print(f"VARYING WIDTHS version {ver}: gates {ag.accuracy_pct:.2f}% | parity {ap.accuracy_pct:.2f}% "
              f"(median {ap.median_err_pct:.2f}%, p95 {ap.p95_err_pct:.2f}%, bias {ap.bias_pct:+.2f}%)", flush=True)

    gf = pd.read_csv(RES / "gate_ml_fixed.csv", dtype={"pattern": str})
    pspf = pd.read_csv(RES / "circuit_spice.csv", dtype={"vector": str})
    pspf = pspf[pspf.corner.isin(gf.corner)]
    ag = summary(gf.assign(all=1), ["all"]).iloc[0]
    nand = gf[gf.gate == "NAND2"].pivot(index="corner", columns="pattern", values="leak_ml")
    pf = parity_sum(nand, cnt, pspf)
    ap = summary(pf.assign(all=1), ["all"]).iloc[0]
    rows.append(row("fixed sizes", VERSIONS["A"], ag, ap))
    print(f"FIXED SIZES (reference) version A: gates {ag.accuracy_pct:.2f}% | parity {ap.accuracy_pct:.2f}% "
          f"(bias {ap.bias_pct:+.2f}%)", flush=True)

    s = pd.DataFrame(rows)
    s.to_csv(RES / "summary.csv", index=False, float_format="%.4g")
    circ.to_csv(RES / "circuit_ml.csv", index=False, float_format="%.6e")
    circ["ones"] = circ.vector.str.count("1")
    by = [summary(circ.rename(columns={f"leak_{v}": "leak_ml"}), ["ones"]).assign(version=v) for v in ("A", "B")]
    pd.concat(by, ignore_index=True).to_csv(RES / "parity_by_ones.csv", index=False, float_format="%.4g")

    tab = []
    for setup, ver, d in (("varying widths", "A", circ.rename(columns={"leak_A": "leak_ml"})),
                          ("varying widths", "B", circ.rename(columns={"leak_B": "leak_ml"})),
                          ("fixed sizes", "A", pf)):
        t = summary(d.assign(circuit="parity8"), ["circuit"]).iloc[0].to_dict()
        tab.append(dict(circuit="parity8", setup=setup, version=VERSIONS[ver], n=t["n"],
                        spice_median_A=d.leak_spice.median(), ml_median_A=d.leak_ml.median(),
                        bias_pct=t["bias_pct"], mean_abs_err_pct=t["mean_abs_err_pct"],
                        median_err_pct=t["median_err_pct"], p95_err_pct=t["p95_err_pct"],
                        accuracy_pct=t["accuracy_pct"]))
    pd.DataFrame(tab).to_csv(RES / "circuit_results.csv", index=False, float_format="%.4g")
    with pd.option_context("display.width", 250):
        print(s.round(3).to_string(index=False))
        print(pd.DataFrame(tab).round(4).to_string(index=False))
    print("STEP 8 DONE", flush=True)


if __name__ == "__main__":
    main()
