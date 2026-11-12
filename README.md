# ML-Driven PVT-Aware Leakage Estimation

This is a Digital VLSI Design course project:
- single-transistor SPICE data trains ML models of the four terminal currents of NMOS and PMOS (ON and OFF);
- those models alone predict the leakage of logic gates and of an 8-bit parity checker;
- every prediction is checked against SPICE.

| Item | Value |
|---|---|
| Model file | `models/models/16nm_HP.pm` (PTM 16 nm HP, BSIM4) |
| Nominal VDD | 0.9 V (sampled 0.81–0.99 V) |
| Channel length / width | L = 16 nm; W = 16–128 nm |
| Test circuit | 8-bit parity checker (7 XOR2 = 28 NAND2, 112 transistors) |
| Software | ngspice-36, Python 3.10, numpy 2.2, pandas 2.3, scikit-learn 1.7, scipy 1.15, joblib 1.6 |

This top-level folder holds the **final method only**. The earlier pipeline, every experiment that led to the final method, and their results are in [`old/`](old/README.md). How the accuracy improved step by step is in `ACCURACY_LOG.md`. The full method and result write-up is in `PROJECT_GUIDE.md`.

## Final results (main setup: varying widths)

The setup:
- the 200 PVT corners each get one random NMOS width Wn and one random PMOS width Wp (16–128 nm);
- every transistor uses its real gate voltage;
- error = (ML − SPICE) / SPICE, and accuracy = 100 % − mean |error|.

| | Gates (18 cases × 200 corners) | 8-bit parity checker (256 vectors × 200 corners) |
|---|---|---|
| **Version A**: node voltages from SPICE | **98.69 %** (median 0.63 %, p95 5.0 %, bias +0.08 %) | **96.91 %** (median 2.50 %, p95 8.2 %, bias −2.29 %) |
| **Version B**: node voltages predicted by the ML models | **98.15 %** (median 1.03 %, p95 6.3 %, bias +0.28 %) | **97.08 %** (median 2.14 %, p95 7.9 %, bias −0.96 %) |
| Reference: fixed gate sizes, version A | 98.93 % | 97.11 % |

There is also an optional extension: a **voltage-aware parity checker**. Each NAND2 input driven by another NAND2 gets that driver's ML output voltage instead of an ideal 0 or VDD. It reaches **98.24 %** (median 1.18 %, bias +0.40 %). It is not the prescribed Part C method, so it is reported only as an improvement.

**Transistor models** (test split, median relative error of Id / Ig / Is / Ib):

| Case | Median Id / Ig / Is / Ib |
|---|---|
| NMOS ON | 0.36 / 0.16 / 0.36 / 0.38 % |
| NMOS OFF | 0.48 / 0.00 / 0.46 / 0.13 % |
| PMOS ON | 0.36 / 0.06 / 0.36 / 0.35 % |
| PMOS OFF | 7.0 / 3.1 / 7.3 / 0.40 % |

Currents of at least 10⁻¹¹ A have a median error of 0.14–0.28 % in all four cases. The large PMOS OFF errors sit where its current falls to the ~10⁻¹⁴ A junction / noise floor, a range the gates do not use.

All tables are in `results/` (`summary.csv`, `gate_results.csv`, `transistor_metrics.csv`, …) and the figures are in `figures/`.

## How to run

```bash
pip3 install --user numpy pandas scikit-learn scipy matplotlib joblib lightgbm
./run_all.sh                 # steps 1-9, then verify_final.py   (~2 h with the shipped data and models)
./run_all.sh 10              # optional voltage-aware parity checker (~3 h)
SMOKE=1 ./run_all.sh 5 6 7 8 9   # quick test of the scripts on 3 corners (results/smoke/)
python3 verify_final.py      # check results against the reported numbers
```

Steps 1, 3, 4, 5 and 7 keep their outputs if those already exist, so the expensive artifacts (SPICE data, model comparison, trained models, SPICE references) are reused. Delete a file to regenerate it with the same seeds; each of those takes hours. Steps 2, 6, 8, 9 and 10 always recompute.

| Step | Script | Output | Deliverable |
|---|---|---|---|
| 1 | `01_gen_transistor_data.py` | `data/<case>.csv` (10,000 rows, split 70/15/15), `data/train_extra/` (80,000 more training rows per case), `netlists/<case>_example.cir` | 1, 2 |
| 2 | `02_data_check.py` | `results/kcl_check.csv` (+ `_train_extra`), `figures/trend_*.png` | 3 |
| 3 | `03_train_models.py` | `results/model_comparison_val.csv` (4 types × 4 cases), `models_trained/<case>.joblib`, `results/transistor_metrics.csv` | 4, 5 |
| 4 | `04_gen_corners.py` | `data/corners.csv` (200 PVT corners), `data/corners_w.csv` (+ Wn, Wp) | |
| 5 | `05_gate_spice.py` | `results/gate_spice_w.csv` (varying widths), `results/gate_spice.csv` (fixed sizes), `netlists/gate_*.cir` | 2 |
| 6 | `06_gate_ml.py` | `results/gate_ml.csv`, `gate_results.csv`, `node_voltages.csv`, `gate_results_fixed.csv` | 6 |
| 7 | `07_circuit_spice.py` | `results/circuit_spice_w.csv`, `results/circuit_spice.csv`, `netlists/parity8*.cir` | 2 |
| 8 | `08_circuit_ml.py` | `results/circuit_ml.csv`, `circuit_results.csv`, `parity_by_ones.csv`, `summary.csv` | 8 |
| 9 | `09_figures.py` | `figures/pred_vs_spice_<case>.png`, `gate_parity.png`, `circuit_parity.png`, `gate_accuracy_by_case.png`, `error_vs_conditions.png` | 5, 7, 8 |
| 10 | `10_vparity_optional.py` | `results/vparity_summary.csv`, `vparity_depth<d>.csv` | (extension) |

Code:
- `final_method.py` has the whole final method: models, device currents with the real gate voltage and corner widths, the ML node solver, the cut-set leakage, and the metrics.
- Shared modules: `config.py` (assignment, ranges, seeds), `spice_utils.py` (model card, batched ngspice), `ml_utils.py` (features, hi/lo frame, KCL, two experts), `circuits.py` (gate and parity netlists, cut-set evaluator).

## Reproducibility

- **The same artifacts as the reported results.** `data/train_extra/`, `models_trained/` and the SPICE references were copied from the experiment runs. `MANIFEST.sha256` holds their SHA-256 hashes. The models were moved from the experiment classes to `final_method.py` without changing any number (`tools/convert_models.py`); `tools/check_converted.py` showed identical predictions on all 3,001 validation and test rows per case.
- **`verify_final.py`** checks two things:
  1. the input hashes;
  2. every result table against `expected_results/`, the outputs of the runs behind the reported numbers (see `expected_results/SOURCES.md`).

  With the shipped artifacts every table is **IDENTICAL**.
- **Bit-for-bit evaluation.** The evaluation steps pin numpy / BLAS to one thread per process, so they give bit-identical numbers on every run.
- **Regeneration from scratch.** SPICE runs are deterministic: in a 3-corner test, regenerated gate and parity references were byte-identical to the originals. Retraining the GPs from scratch can differ in the last floating-point digits (optimizer and BLAS details). Use `verify_final.py --tol` to see how close a from-scratch run is.
- **Large files.** `data/train_extra/` (84 MB) and `models_trained/` (145 MB) are not in git. Regenerate them with steps 1 and 3, or copy them from the project machine and check them against `MANIFEST.sha256`.

## Method in brief

**Part A: data.**
- VDD, T, W, Vd, Vs and the five process parameters (vth0, toxe, ndep, u0, xj; ±10 %) are drawn uniformly and independently.
- Vg and Vb are set by the case: NMOS ON Vg = VDD, Vb = 0; NMOS OFF Vg = 0, Vb = 0; PMOS ON Vg = 0, Vb = VDD; PMOS OFF Vg = VDD, Vb = VDD.
- 10,000 rows per case are split 70/15/15.
- 80,000 more training rows per case come from the same generator with new seeds.

**Part A: models.**
- **Frame and target.** Symmetric hi/lo frame; target ln(|I|/W).
- **KCL.** The one mixed-sign current is computed from KCL, so predictions satisfy KCL exactly.
- **Features.** Thermal-voltage-scaled physics features; the ON cases also use plain Vgs, Vgd, Vds, because gate tunnelling depends on the oxide voltage.
- **OFF cases.** Two experts (channel-dominated vs gate/junction-dominated).
- **Regressor.** An averaged Gaussian process on 87,000 rows: kernel fitted on 2,000 points, then 9 exact GPs on disjoint slices of ≤ 9,700 rows, averaged.
- **Selection.** Chosen on the validation split; the test split is scored once.

**Part B: gates.**
- Gates: INV, NAND2, NOR2, AND2 = NAND2 + INV, OR2 = NOR2 + INV.
- **Device currents.** Each transistor's ON/OFF state comes from its gate's logic level, and its currents come from that case's model. The model is evaluated at the transistor's terminal voltages, its real gate voltage and its corner width.
- **Leakage (cut-set).** Nodes are split into logic-1 and logic-0 sets; by KCL the leakage is the current crossing between them. Devices with drain and source on the same side contribute only gate and body current; OFF devices across the split contribute their terminal currents.
- **Node voltages.** Version A takes them from SPICE. Version B solves them from KCL with the models: bisection plus Gauss-Seidel, starting from the logic levels.

**Part C: parity checker.**
- A logic simulation gives each NAND2's input pattern.
- The circuit leakage is the sum of the 28 NAND2 leakages.

SPICE gate and circuit results are used only for evaluation.
