# Remaining work

Status of the project and what is still to do, in priority order.

## Current status

| Level | Median error | Mean accuracy | Bias |
|---|---|---|---|
| Transistor models (NMOS ON / NMOS OFF / PMOS ON) | 0.38–0.49 % | – | – |
| Transistor model (PMOS OFF) | 6.1 % overall, ~1 % in the region gates use | – | – |
| Gates (18 cases × 200 corners) | 2.2 % | 95.6 % | −0.2 % |
| 8-bit parity checker (256 vectors × 200 corners) | 4.1 % | 95.3 % | −3.7 % |

Deliverables 1–9 (datasets, netlists, data check, model comparison, chosen models, gate table and parity plot, test-circuit results, code) are in the repo. The report (deliverable 10) is not written yet.

---

## Must do (required by the spec)

### 1. Write the report (deliverable 10)

| Section | Status |
|---|---|
| Setup: model file, VDD, test circuit, simulator and version (ngspice-36) | Ready |
| Data generation: sampling, netlists, train/val/test split, data check | Ready |
| Models: 4 types, hyperparameters, preprocessing, comparison table, why GP | Ready |
| Gate results: table, parity plot, best and worst cases | Ready |
| Test-circuit results compared with gate-level accuracy | Ready |
| Error analysis: largest errors by case, gate, input and **corner conditions (high T, low vth0, ...)** | Needs task 2 |
| Limitations and improvements | Ready |

Format still to decide (LaTeX or Word, figures embedded). Check whether the course requires one.

### 2. Error-analysis script
- [ ] Plot and tabulate gate and parity errors against temperature, vth0, VDD, toxe and the other corner parameters.
- [ ] Find where the largest errors occur (case, gate, input pattern, corner) and explain why.

### 3. Visually check all figures
- [ ] Re-check the final plots after the last retrain: `gate_parity.png`, `circuit_parity.png`, `pred_vs_spice_*.png`, `trend_*.png`.
- [ ] Make sure each is readable, labelled and fit for the report.

### 4. Clean end-to-end rerun of `./run_all.sh`
- [ ] Deliverable 9 requires the scripts to regenerate everything in order from the model file. Steps 1, 2, 4, 5 and 7 last ran with earlier code versions; a fresh full run proves reproducibility.
- About 2.5 hours, mostly the parity-checker SPICE. Runs unattended on the Linux machine (`nohup ./run_all.sh > results/run_all.log 2>&1 < /dev/null &`).

---

## Should do

### 5. Confirm assumptions with the TA
- [ ] Gate transistor widths: unit 24 nm (1.5·Lmin); INV 1/2, NAND2 2/2, NOR2 1/4 units.
- [ ] Leakage definition: current drawn from VDD **plus** inputs held at logic 1.
- [ ] Meaning of the "99 % mean accuracy" target (currently 100 % − mean |relative error|).
- [ ] Parity-checker structure: 7 XOR2, each built from 4 NAND2 (28 NAND2), no output inverter.

If any answer differs, the affected steps (5–8, possibly 3) need rerunning, about 1–2 hours.

### 6. Dataset-size study
- [ ] Train with 2k, 4k, 7k and 10k samples and plot the error against N, to justify the dataset size. The spec asks for "the smallest dataset possible without compromising accuracy".
- About 1 hour on the Linux machine.

### 7. Update `PROJECT_PLAN.md`
- [ ] It still describes the original plan. Update it with the decisions actually made, or point it to the README.

---

## Optional (higher accuracy)

### 8. Improve PMOS OFF at full Vds
- This causes the −2.4 % bias in PMOS-dominated gates (INV `1`, NAND2 `11`, AND2, OR2 `00`) and most of the −3.7 % parity-checker bias.
- The most promising lever is more PMOS OFF training data. **Needs approval first:** the deliverables list asks for 10,000-row CSVs.
- Expected gain is uncertain, roughly 95 % → 97 % accuracy.

### 9. Parity-checker modelling gap
- Part of the extra circuit-level error comes from assuming ideal 0/VDD gate inputs and no inter-gate loading. The spec itself says to expect this.
- Explain it in the report rather than fix it.

---

## Suggested order
1. Error-analysis script and figure review (tasks 2, 3).
2. Start the clean full rerun (task 4) in the background on the Linux machine.
3. Write the report (task 1) while the rerun runs.
4. Dataset-size study (task 6) if time allows.
5. Ask the TA the questions in task 5 as early as possible.

## Open questions for the team
- Submission deadline?
- Required report format?
