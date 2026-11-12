# old/: earlier pipeline and all experiments (archive)

This folder is a snapshot of the project before the final method was moved to the top level. Nothing here is needed to reproduce the final results; see `../README.md` for that.

| Folder / file | What it is |
|---|---|
| `01_…08_*.py`, `run_all.sh`, `results/`, `models_trained/`, `figures/`, `netlists/` | first complete pipeline: 7,000-row GP models, fixed gate sizes (gates 95.6 %, parity 95.3 %) |
| `exp_extra/` | 40,000 extra training rows per case; GP_cap / MLP_ens candidates |
| `exp_acc/` | 40,000 more rows; averaged GP (GPBag87), plain-voltage features for ON (GPX), MLP comparisons, size study, fresh-corner check, version B (ML node voltages) |
| `exp_width/` | varying widths: per-corner Wn / Wp, new gate and parity SPICE |
| `exp_vg/` | real gate voltage of each transistor (final method), and the rejected off-rail retraining experiment |
| `exp_vparity/` | voltage-aware parity checker (optional extension) |
| `config.py`, `circuits.py`, `ml_utils.py`, `spice_utils.py` | copies of the shared modules (identical to the top level), so the code here still runs |
| `data`, `models` | links to `../data`, `../models` (same files) |

The scripts run from this folder as before, e.g. `cd old && ./run_all.sh` or `cd old/exp_vg && python3 eval_vg.py`. Large generated files (trained models, extra data) are not in git. They stay on the project machine.

The step-by-step history of the accuracy is in `../ACCURACY_LOG.md`.

---

*Original README of the first pipeline:*

# ML-Driven PVT-Aware Leakage Estimation

Digital VLSI Design course project. Single-transistor SPICE data trains ML models of the four terminal currents. Those models then predict the leakage of logic gates and of an 8-bit parity checker, and every prediction is checked against SPICE.

| Item | Value |
|---|---|
| Model file | `models/models/16nm_HP.pm` (PTM 16 nm HP, BSIM4 level 54) |
| Nominal VDD | 0.9 V (sampled 0.81–0.99 V) |
| Channel length | 16 nm; W = 16–128 nm |
| Test circuit | 8-bit parity checker (7 XOR2 = 28 NAND2, 112 transistors) |
| Simulator | ngspice-36 (Ubuntu 22.04), Python 3.10, scikit-learn 1.7 |

## How to run

```bash
pip3 install --user numpy pandas scikit-learn scipy matplotlib joblib lightgbm
./run_all.sh            # steps 1-8 in order, fixed seeds; logs in results/log_<step>.txt
./run_all.sh 3 6 8      # only some steps
```

| Step | Script | Output |
|---|---|---|
| 1 | `01_gen_transistor_data.py` | `data/{nmos,pmos}_{on,off}.csv` (10,000 rows each), `netlists/*_example.cir` |
| 2 | `02_data_check.py` | `results/kcl_check.csv`, `figures/trend_*.png` |
| 3 | `03_train_models.py` | `results/model_comparison_val.csv`, `results/chosen_models_test.csv`, `models_trained/*.joblib`, `figures/pred_vs_spice_*.png` |
| 4 | `04_gen_corners.py` | `data/corners.csv` (200 PVT corners) |
| 5 | `05_gate_spice.py` | `results/gate_spice.csv`, `netlists/gate_*.cir` |
| 6 | `06_gate_ml.py` | `results/gate_ml.csv`, `results/gate_results.csv`, `results/gate_node_voltages.csv`, `figures/gate_parity.png` |
| 7 | `07_circuit_spice.py` | `results/circuit_spice.csv`, `netlists/parity8.cir` |
| 8 | `08_circuit_ml.py` | `results/circuit_ml.csv`, `results/circuit_results.csv`, `figures/circuit_parity.png` |

Shared code: `config.py` (assignment, ranges, seeds), `spice_utils.py` (model-card editing, batched ngspice), `ml_utils.py` (features, models), `circuits.py` (gate / parity netlists, ML gate-leakage evaluator).

## Method

### Data (Part A)
- **Sampling.** VDD is drawn first. Then Vd and Vs are drawn uniformly in [0, VDD], plus W, T and the five process parameters (vth0, toxe, ndep, u0, xj, each nominal ±10 %, separately for NMOS and PMOS), all uniform and independent.
- **Netlists.** Each sample gets its own model card with the parameters substituted, and one voltage source per terminal. Netlists include `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9`. One ngspice process runs hundreds of circuits, and four processes run in parallel.
- **Sign convention.** Currents are stored as the raw ngspice source currents (positive = out of the device terminal), the same convention as the TA's sample database.

### Transistor models
- **Symmetric frame.** The higher of (Vd, Vs) is called "hi". In this frame almost every current has a fixed sign.
- **Kirchhoff's law.** The one current per case whose sign still varies is not learned; it is computed as −(sum of the other three). Predictions therefore satisfy KCL exactly.
- **Target and inputs.** The target is ln(|I| / W). The inputs are thermal-voltage-scaled biases (Vgs/Vt, Vgd/Vt, Vds/Vt, ln(1 − e^(−Vds/Vt)), |vth0|/Vt) plus the process parameters.
- **Model types compared.** Polynomial ridge (degree 3), LightGBM, MLP (3×64, tanh) and an exact Gaussian process (ARD RBF kernel with a linear mean). The best model per case is chosen on the validation set.
- **Noise floor.** ngspice resolves these currents only in steps of ~2×10⁻¹⁶ A, so currents below 3×10⁻¹⁵ A are numerical noise. The GP ignores them.
- **Two experts for OFF devices.** An OFF transistor's current is channel (sub-threshold) dominated when its source is near its rail. Further away, gate tunnelling and junction currents dominate. In log-current the change-over is a sharp bend. Each OFF model therefore uses two experts, split at the channel strength where the channel/floor current ratio falls below 10; that point is found from the training data (x* = 0.30 for PMOS, 0.125 for NMOS).

### Gates (Part B)
- **Sizing.** Unit width 24 nm (1.5·Lmin). INV n/p = 1/2, NAND2 = 2/2, NOR2 = 1/4 units, following the TA's AND2 example. Every device stays inside the trained W range.
- **SPICE leakage.** I_leak = −(I(Vdd) + Σ I(Vin) for inputs at 1), i.e. the current drawn from everything at logic 1.
- **ML leakage.** The SPICE node voltages give each transistor's terminal voltages. Each device's gate level picks its ON or OFF model. Nodes are split into a logic-1 set and a logic-0 set, and by KCL the leakage is the current crossing between them:
  - Devices whose drain and source are on the same side (ON devices) contribute only their gate and body currents.
  - OFF devices across the split contribute their terminal currents on the logic-1 side.

  This is the rigorous form of "sub-threshold + gate + body leakage" and avoids ON-device channel currents at Vds ≈ mV.
- **ML node voltages.** Internal nodes are also solved from KCL with the ML models (`gate_node_voltages.csv`).

### Parity checker (Part C)
- **SPICE.** One netlist per corner; all 256 input vectors are applied with `alter`.
- **ML.** A logic simulation gives each NAND2's input pattern. Its Part B ML leakage at that corner is looked up, and the 28 values are summed.

## Results

Error = (ML − SPICE) / SPICE; bias = median signed error; accuracy = 100 % − mean |error|.

### Data check
The four terminal currents sum to zero within 1.3×10⁻¹⁵ A in the worst case (ngspice's current resolution). All three trend plots follow device physics: OFF current rises with T and falls exponentially with vth0, and gate current falls exponentially with toxe.

### Transistor models (test set, chosen model = GP in every case)

| Case | Id median / p95 | Ig median / p95 | Is median / p95 | Ib median / p95 | R² (log\|Id\|) |
|---|---|---|---|---|---|
| NMOS ON | 0.54 % / 3.9 % | 0.41 % / 5.7 % | 0.53 % / 4.0 % | 0.48 % / 2.7 % | 0.999 |
| NMOS OFF | 0.67 % / 8.2 % | 0.00 % / 0.04 % | 0.65 % / 10 % | 0.19 % / 22 % | 0.998 |
| PMOS ON | 0.48 % / 3.5 % | 0.11 % / 1.2 % | 0.49 % / 3.3 % | 0.42 % / 2.7 % | 0.998 |
| PMOS OFF | 10 % / 262 % | 2.5 % / 69 % | 11 % / 1275 % | 0.47 % / 18 % | 0.959 |

The PMOS OFF errors are concentrated where its current falls to the 10⁻¹⁴ A junction / noise floor. In the channel-dominated region that gates use, the median error is about 1 %. The full 4 × 4 comparison table is in `results/model_comparison_val.csv`.

### Gates (18 cases × 200 corners), `results/gate_results.csv`
- **All 3,600 simulations:** median error 2.2 %, p95 10 %, bias −0.2 %, mean accuracy 95.6 %.
- **Best cases:** NAND2 `00` (0.48 %), NAND2 `10` (0.64 %), NOR2 `10` (0.80 %), NAND2 `01` (0.84 %).
- **Weakest cases:** those where an OFF PMOS holds the full VDD (INV `1`, NAND2 `11`, AND2 `xx`, OR2 `00`), at about 3.5 % with −2.4 % bias.
- **Internal node voltages:** ML node voltages solved from KCL agree with SPICE within 0.003–1.2 mV (median per case).

### 8-bit parity checker (256 vectors × 200 corners), `results/circuit_results.csv`
- **All 51,200 simulations:** median error 4.1 %, p95 11 %, bias −3.7 %, mean accuracy 95.3 %.
- **Effect of the gate-level assumptions:** ideal 0/VDD inputs and no inter-gate loading add about 1.3 % of bias on top of the NAND2 gate-level error.
