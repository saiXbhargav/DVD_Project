# ML-Driven PVT-Aware Leakage Estimation: Project Plan

Course: Digital VLSI Design. Source: `DVD_ML_Project.pdf`.

**Goal.** Train ML models of single NMOS/PMOS terminal currents. Use them to predict the leakage of logic gates, then of a larger test circuit, under process, voltage and temperature (PVT) variation. Check every prediction against SPICE. The target is **up to 99 % mean accuracy**.

---

## 0. Our assignment

| Item | Value |
|---|---|
| Technology node / model file | **16 nm HP**: `models/models/16nm_HP.pm` (BSIM4, `level = 54`) |
| Nominal VDD | **0.9 V**, so the VDD range is 0.81 – 0.99 V (±10 %) |
| Channel length L | **16 nm** |
| Width range | W ∈ [16 nm, 128 nm] (1–8 × Lmin), for both NMOS and PMOS |
| Test circuit | **8-bit parity checker** |
| Platform | **Linux** (the project folder is on a network drive mounted from the Linux machine) |
| Simulator + version | ngspice, installed on the Linux machine (record the output of `ngspice -v`) |

> Use **only** the assigned model file, VDD and test circuit.

### Nominal process parameters from `16nm_HP.pm` (each sampled at ±10 %)

| Parameter | NMOS nominal | NMOS range | PMOS nominal | PMOS range |
|---|---|---|---|---|
| vth0 | 0.47965 V | 0.43169 – 0.52762 | −0.43121 V | −0.38809 – −0.47433 |
| toxe | 9.5e-10 m | 8.55e-10 – 1.045e-9 | 1.0e-9 m | 9.0e-10 – 1.1e-9 |
| ndep | 7.0e18 | 6.3e18 – 7.7e18 | 5.5e18 | 4.95e18 – 6.05e18 |
| u0 | 0.03 | 0.027 – 0.033 | 0.006 | 0.0054 – 0.0066 |
| xj | 5.0e-9 m | 4.5e-9 – 5.5e-9 | 5.0e-9 m | 4.5e-9 – 5.5e-9 |

---

## 1. Big picture

```
Part A  single-FET SPICE (4 cases x 10k samples)  ->  4 CSV datasets  ->  train 4 ML models
                                                                              |
Part B  5 gates x all inputs x 200 corners:  SPICE leakage  vs  ML leakage (built from the transistor models)
                                                                              |
Part C  test circuit x all inputs x the same 200 corners:  SPICE  vs  sum of ML gate leakages
```

The four transistor cases:

| Case | Vg | Vb | Vd, Vs |
|---|---|---|---|
| NMOS ON | VDD | 0 | U(0, VDD) each, independent |
| NMOS OFF | 0 | 0 | U(0, VDD) each |
| PMOS ON | 0 | VDD | U(0, VDD) each |
| PMOS OFF | VDD | VDD | U(0, VDD) each |

---

## 2. Global rules (apply everywhere)

- [ ] **Every** netlist (transistor, gate and test circuit) contains exactly:
  `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9`
- [ ] In single-transistor netlists, **every terminal has its own voltage source** (Vd, Vg, Vs, Vb). This lets you measure Id, Ig, Is and Ib separately.
- [ ] Fix the random seeds for sampling, data splits and model training. Store them in one config file.
- [ ] Scripts must regenerate everything from the model file when run in order.
- [ ] SPICE results for gates and the test circuit are for **evaluation only**. Never train on them.

> ⚠️ The provided example `22n_NMOS_ON.net` does **not** meet these rules. It uses `GMIN=1e-020`, has no `reltol`/`vntol`, uses a hard-coded Linux `.INCLUDE` path, and sweeps VDD to 1.2 V. Use it only as a structural reference.

---

## 3. Part A: Transistor models

### A1. Sampling (per case, initial N = 10,000)
Draw every parameter independently. **Draw VDD first.**

| Parameter | Distribution |
|---|---|
| VDD | U(0.9·VDDnom, 1.1·VDDnom) |
| Vd | U(0, VDD) |
| Vs | U(0, VDD) |
| W | U(1·Lmin, 8·Lmin) |
| Temp | U(−40, 125) °C |
| vth0, toxe, ndep, u0, xj | U(0.9·nom, 1.1·nom), with nominals read from the `.pm` file. NMOS and PMOS have separate nominals. |

Notes:
- PMOS `vth0` is negative (e.g. −0.4606), so scale it by a factor in [0.9, 1.1]. The sign is unchanged.
- Vg and Vb come from the case table, so they track the sampled VDD.
- N can be reduced or increased. The aim is the **smallest dataset that still gives high accuracy on unseen data**. A learning-curve study is a good way to justify your choice: train on 2k, 4k, 7k and so on, then plot the error.

### A2. Netlist and simulation
- One template netlist per case. Fill in W, L, the terminal voltages, the temperature (`.temp`) and the five process parameters.
- Two ways to inject the process parameters:
  1. write a modified copy of the model card for each sample, or
  2. use ngspice `altermod` / `.param` expressions inside a `.control` loop. This is much faster because many samples run in one ngspice call.
- Run ngspice in batch mode (`ngspice -b`). Parallelise across CPU cores, since there are 40k+ simulations in total.
- Read `I(Vd)`, `I(Vg)`, `I(Vs)` and `I(Vb)`. **Pick one sign convention** and document it. ngspice reports source current flowing into the + node through the source, so the current *into* the device terminal is usually `−I(Vx)`.
- The model card already sets `igcmod=1` and `igbmod=1`, so gate tunnelling current is modelled and Ig ≠ 0.

### A3. Dataset files (deliverable 1)
Create four CSVs, e.g. `nmos_on.csv`, `nmos_off.csv`, `pmos_on.csv` and `pmos_off.csv`. Use exactly these columns:
```
vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj, id, ig, is, ib
```

### A4. Data check (deliverable 3)
- [ ] **Kirchhoff check** for each case: `id + ig + is + ib ≈ 0`. Report the max and median residual, both absolute and relative to max(|I|).
- [ ] Plot: OFF current vs temperature (expect an exponential rise).
- [ ] Plot: OFF current vs vth0 (expect an exponential fall as vth0 rises).
- [ ] Plot: gate current vs toxe (expect a strong fall as toxe rises).

### A5. Training (deliverables 4, 5)
- Split with a fixed seed: **70/15/15** (7000/1500/1500). An 80/10/10 split is also allowed.
- Train **4 model types** for **each of the 4 cases**, for example:
  1. Polynomial / Ridge regression (baseline)
  2. Random Forest
  3. Gradient boosting (XGBoost / LightGBM)
  4. Neural network (MLP)
- Each model predicts all four terminal currents (Id, Ig, Is, Ib). Use one multi-output model or one model per terminal, and say which in the report.
- **Preprocessing (important).** Currents span many decades and can change sign:
  - Target: `log10(|I|)`, plus separate handling of the sign. The sign is often deterministic from Vd vs Vs; otherwise use a small classifier. A signed transform such as `asinh(I/I0)` is an alternative.
  - Useful engineered features: `vds = vd − vs`, `vgs`, `vbs`, `1/T(K)`, `w/L`.
  - Standardise the features for Ridge and the MLP.
- **Metrics** (on currents, not on logs): relative error `e = (I_pred − I_spice)/I_spice`.
  - R² (on the log target, or state which scale you use), **median |e|** and **95th-percentile |e|**.
- **Model comparison table** (deliverable 4): 4 models × 4 cases, with R², median error and P95 error on the **validation** set.
- Choose the best model per case **using validation**, then report the **test-set** median and P95 error.
- [ ] Predicted-vs-SPICE plot for each case (log-log).
- [ ] Save the 4 chosen models (e.g. `models_trained/nmos_on.pkl`).

---

## 4. Part B: Gate-level validation

### B1. Corners (shared with Part C)
- Generate **200 corners once** with a fixed seed and save them to `corners.csv`.
- A corner is one draw of VDD, temperature, the **NMOS** process parameter set and the **PMOS** process parameter set. All NMOS in a gate share one draw, and all PMOS share another.
- Inputs are applied at exactly 0 or VDD.
- Transistor widths for the gates are not given in the spec. Choose fixed sizes inside [1, 8]·Lmin (e.g. Wn = 2L, Wp = 4L), state them in the report, and use the same values in SPICE and ML.

### B2. Gates and cases (18 cases × 200 corners = 3600 SPICE runs)

| Gate | Inputs | Internal nodes to record |
|---|---|---|
| INV | A = 0, 1 | none |
| NAND2 | AB = 00, 01, 10, 11 | x (between the series NMOS) |
| NOR2 | AB = 00, 01, 10, 11 | y (between the series PMOS) |
| AND2 | AB = 00, 01, 10, 11 | x and the NAND2 output (NAND2 → INV) |
| OR2 | AB = 00, 01, 10, 11 | y and the NOR2 output (NOR2 → INV) |

- Simulate AND2 and OR2 as **complete** netlists. Do not add separate NAND and INV results.
- From each SPICE run, record the leakage, the internal node voltages and the output voltage.

### B3. Leakage definition (decide once, use everywhere)
The total current drawn from the supply side in steady state is recommended. To account for the gate current from inputs held at VDD, use **P_static / VDD = Σ over sources of (V·I)/VDD**. Alternatively, use only `I(VDD)`. Whichever you pick, use the **same definition** in SPICE and ML and state it in the report.

### B4. ML gate prediction
For every gate × input × corner:
1. Take the **SPICE-recorded** internal node voltages and output voltage.
2. Work out (Vd, Vg, Vs, Vb) for each transistor. Example for NAND2 (top NMOS = A, bottom NMOS = B):

   | Device | d | g | s | b | Model used |
   |---|---|---|---|---|---|
   | PMOS-A | out | A | VDD | VDD | A=0 → PMOS ON, A=1 → PMOS OFF |
   | PMOS-B | out | B | VDD | VDD | B=0 → PMOS ON, B=1 → PMOS OFF |
   | NMOS-A | out | A | x | 0 | A=1 → NMOS ON, A=0 → NMOS OFF |
   | NMOS-B | x | B | 0 | 0 | B=1 → NMOS ON, B=0 → NMOS OFF |

   The gate voltage picks the model: NMOS with g=VDD uses the ON model and g=0 the OFF model. PMOS with g=0 uses the ON model and g=VDD the OFF model. The body always matches the training case.
3. Feed the terminal voltages, the corner's VDD/T/process parameters and W into the right model. Get Id, Ig, Is and Ib for each device.
4. Combine the device currents into gate leakage using the B3 definition (sum the terminal currents at VDD-connected nodes and inputs at 1).
5. *(Optional, makes the result stronger.)* Solve the internal node voltages with the ML models by KCL root-finding at x/y. Report these as the "ML internal node voltages" next to the SPICE ones. The deliverable asks for node voltages "from both".

Watch out: devices that are ON have Vds ≈ 0 in the leakage state, so their currents are tiny and dominated by gate tunnelling. Make sure the ON models are accurate near Vds = 0.

### B5. Results (deliverables 6, 7)
- [ ] Table with 18 rows, one per gate × input. Columns: **SPICE median, ML median, bias, median |error|, P95 |error|**.
  - bias = median (or mean) of the signed relative error. Say which.
- [ ] One **parity plot**, ML vs SPICE leakage, log–log, all 3600 points, coloured by gate.
- [ ] Comment on the most and least accurate gates and input combinations.

---

## 5. Part C: Test-circuit validation

### C1. Circuit: 8-bit parity checker (our assignment)
Only INV, NAND2, NOR2, AND2 and OR2 have models, so **build the circuit from these gates only**:
- Parity output P = D0 ⊕ D1 ⊕ … ⊕ D7: **7 XOR2 gates** in a balanced tree (4 → 2 → 1).
- Each XOR2 is built from **4 NAND2**: n1 = NAND(a,b), n2 = NAND(a,n1), n3 = NAND(b,n1), out = NAND(n2,n3). That gives **28 NAND2** and 112 transistors in total.
  - Another option is XOR built from NOR2/AND2/OR2. Whichever you choose, make the SPICE netlist and the ML gate list describe exactly the same structure.
  - For even parity, add an INV at the output and include it in both SPICE and ML.
- There are 2⁸ = **256 input vectors**, giving 256 × 200 corners = **51,200 SPICE operating points**. Batch them, for example one ngspice run per corner that loops over the 256 vectors with `alter`.
- Inside each XOR, NAND inputs are driven by other NAND outputs. In SPICE these sit near, but not exactly at, 0/VDD. The ML side assumes ideal levels, which is one source of the expected accuracy drop.
- Use the same transistor sizes as Part B.

### C2. SPICE reference
- [ ] One complete transistor-level netlist with the required `.options`.
- [ ] Simulate every input vector at the **same 200 corners**. Record the total leakage with the B3 definition.

### C3. Hierarchical ML prediction
1. For each input vector, run a **logic simulation** to get the logic value at every gate's inputs.
2. For each gate instance, call the **gate model** from Part B: (gate type, input combination, corner) → ML leakage.
3. Sum the gate leakages to get the circuit leakage.

This assumes ideal 0/VDD levels at gate inputs and ignores loading, so accuracy is expected to be a little lower than at gate level.

### C4. Results (deliverable 8)
- [ ] The same metrics table as B5 (overall, and optionally per input vector or grouped).
- [ ] Parity plot, ML vs SPICE, log–log.
- [ ] Compare against gate-level accuracy.

---

## 6. Suggested repo layout and script order

```
DVD/
├── config.py                  # seeds, assigned model file, VDD, L, W ranges, gate sizes, paths
├── 01_gen_transistor_data.py  # Part A sampling + ngspice runs -> data/*.csv
├── 02_data_check.py           # KCL check + 3 trend plots
├── 03_train_models.py         # 4 models x 4 cases, comparison table, test metrics, save models
├── 04_gen_corners.py          # 200 corners -> data/corners.csv
├── 05_gate_spice.py           # 18 cases x 200 corners -> results/gate_spice.csv
├── 06_gate_ml.py              # ML gate leakage -> results/gate_ml.csv, table, parity plot
├── 07_circuit_spice.py        # test circuit SPICE -> results/circuit_spice.csv
├── 08_circuit_ml.py           # hierarchical ML -> table, parity plot
├── netlists/                  # templates + example netlists (deliverable 2)
├── data/  models_trained/  results/  figures/
└── report/                    # typed report (deliverable 10)
```

---

## 7. Deliverables checklist

1. [ ] **Datasets:** 4 CSVs (10,000 rows each, or the justified N) with the exact columns listed in A3.
2. [ ] **Example netlists:** 1 per transistor case (4), 1 per gate (5), and the test-circuit netlist.
3. [ ] **Data check:** KCL check per case and 3 trend plots.
4. [ ] **Model comparison table:** 4 models × 4 cases with R², median and P95 error on validation.
5. [ ] **Chosen models:** test median and P95 error, 4 predicted-vs-SPICE plots, 4 saved model files.
6. [ ] **Gate results table:** 18 rows with SPICE median, ML median, bias, median error and P95 error.
7. [ ] **Gate parity plot:** log axes, all gate simulations.
8. [ ] **Test-circuit results:** table and log–log parity plot.
9. [ ] **Code:** ordered scripts with fixed seeds that regenerate everything from the model file.
10. [ ] **Report** (see §8).

---

## 8. Report outline

1. **Setup:** model file, nominal VDD, test circuit, simulator and version, gate transistor sizes.
2. **Data generation:** sampling scheme, netlist construction and parameter injection, split method and seed, data-check results.
3. **Models:** the 4 model types, their main hyperparameters, preprocessing (log|I|, sign handling, features), the comparison table, and why each final model was chosen.
4. **Gate results:** table and parity plot, with the best and worst gates and inputs.
5. **Test-circuit results:** metrics and parity plot compared with gate-level accuracy.
6. **Error analysis:** where the largest errors are (case, gate, input, or corners such as high T or low vth0) and why. Likely causes: stacking effect / intermediate node voltages, very small ON-device currents near Vds ≈ 0, extreme corners at the edges of the training distribution, and gate tunnelling dominance at thin toxe.
7. **Limitations and improvements:** ideal input levels, ignored inter-gate loading, model extrapolation at corner extremes, dataset size vs accuracy. Possible fixes: denser sampling near Vds ≈ 0, physics-informed features, ensembles, KCL-solved internal nodes.

---

## 9. Open points to confirm with the TA

- Exact **leakage definition**: `I(VDD)` only, or including input-source currents.
- **Transistor widths** for gates and the test circuit.
- Definition of **"accuracy"** for the 99 % target, e.g. `100 − mean|relative error|%`.
- Whether the "ML internal node voltages" must come from a KCL solve with the ML models (§B4 step 5).

## 10. Environment notes

- **Platform: Linux.** This folder is a network drive mounted from the Linux machine. Run all simulations and scripts on that machine.
- Install ngspice there, e.g. `sudo apt install ngspice` (or build from source / use `conda install -c conda-forge ngspice`). Record the output of `ngspice -v` in the report.
- Python stack: numpy, pandas, scikit-learn, xgboost/lightgbm, (torch or sklearn MLP), matplotlib, joblib.
- In netlists, use **relative** `.INCLUDE` paths (e.g. `.include ../models/models/16nm_HP.pm`) so scripts run from the repo.
- `16nm_HP.pm` is BSIM4 (`level = 54`) with `igcmod=1` and `igbmod=1`, so gate leakage is modelled.
