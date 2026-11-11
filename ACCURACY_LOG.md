# Accuracy improvement log

Running record of everything done to raise the accuracy after the first complete
pipeline (the "official" results in README.md / PROJECT_GUIDE.md), step by step and
case by case. Last update: 10 Nov 2026 (all runs finished).

**Where we are (official corners, accuracy = 100 − mean error):**

| | Start (official) | Now (GPX) | Fresh unseen data (GPX) |
|---|---|---|---|
| Gates | 95.58 % | **98.65 %** | 98.61 % |
| Parity checker | 95.33 % | **97.11 %** | 97.14 % |

**Ground rules followed in every step**

- Only our own data: every extra training row was simulated by us with the same generator,
  distribution, model card and SPICE options as step 1 (`01_gen_transistor_data.py`), only
  with new, disjoint seeds.
- No existing pipeline file was modified. All experiments live in `exp_extra/` and `exp_acc/`
  and only import the project modules.
- Choices between options are made on validation data only: the original transistor
  validation split and, from round 1 on, 100 new **validation corners** for the gates.
  The official 200 corners (gates + parity checker) are used only to report results.
- The gate and parity ML predictions use only the four transistor models (SPICE gate /
  circuit results are references only; gate internal node voltages come from SPICE as the
  project document prescribes).
- Accuracy = 100 − mean |error|, error = (ML − SPICE) / SPICE; bias = median signed error.

---

## 0. Starting point (official pipeline)

| Part | Result |
|---|---|
| Transistors (test, mean of the four per-current medians) | NMOS ON 0.49 %, NMOS OFF 0.38 %, PMOS ON 0.38 %, PMOS OFF 6.08 % |
| Gates (18 cases × 200 corners) | **95.58 %**, median 2.16 %, p95 10.34 %, bias −0.23 % |
| Parity checker (256 vectors × 200 corners) | **95.33 %**, median 4.08 %, p95 11.24 %, bias −3.67 % |

Model: exact GP (ARD RBF + linear mean) on 7,000 training rows per case, physics
features scaled by kT/q, hi/lo frame, KCL-derived mixed-sign current, two experts for OFF.

## 1. Data verification (before adding anything)

`scratch_datacheck.py`, `scratch_datacheck2.py` (read-only) on all 200,000 transistor rows,
the corners and the gate / parity SPICE results:

- rows, columns, no NaN, no duplicates; VDD 0.81–0.99 V, T −40…125 °C, W 16–128 nm
  (1–8 × Lmin), 0 ≤ Vd, Vs ≤ VDD, Vg / Vb rules, process parameters within ±10 %: all pass;
- original and extra data have the same input distribution (KS test, all p > 0.05);
- re-simulating random rows reproduces the stored currents (< 2e-8 relative above 1 pA);
- KCL residual ≤ 1e-14 A (solver tolerance; < 0.08 % of any current above 10 pA);
- gates: every output at its logic level, supply vs ground current within 2e-5;
- the flagged "sign" and "trend" checks were physics, not errors: in OFF devices with the
  source far from the rail, gate / junction current exceeds the pinched-off channel current;
  PMOS vth0 is negative, so a "higher" vth0 means more current.

Conclusion: the data is correct; nothing had to be regenerated.

## 2. Experiment 1 — more training rows (`exp_extra/`)

- 40,000 extra rows per case (4 blocks × 10k, seeds 5001–5304) → pool of 47,000.
- Exact GP cost grows as n³ (7 GB machine), so **GP_cap** = GP on 9,000 rows drawn from the pool.
- Also tried an ensemble of 3 MLPs (3 × 128) on all 47k rows.

| | Official GP 7k | **GP_cap** | MLP ensemble 47k |
|---|---|---|---|
| Gates | 95.58 % | **97.68 %** | 94.48 % |
| Parity | 95.33 % | **96.86 %** | 93.23 % |

GP_cap removed most of the systematic bias at the "OFF transistor holding the full VDD"
operating point (INV 1, NAND2 11, AND2: −2.35 % → ~0 %; INV 0, NOR2 00: +1.07 % → −0.55 %).
The MLP ensemble was undertrained (early stopping after 30 epochs without improvement).

## 3. Round 1 — all data, averaged GPs, validation corners (`exp_acc/`, run 9 Nov 13:22–15:17)

What was added:

1. **40,000 more rows per case** (seeds 7001–7304, ~25 s per 10k block) → pool of 87,000.
2. **100 validation corners** (seed 4321) with gate SPICE (1,800 simulations, none failed),
   used only to choose between options.
3. **Averaged GP (GPBag)**: kernel hyper-parameters fitted once per output, then one exact GP
   per disjoint slice of ≤ 9,700 rows; prediction = linear mean + average of the slice GPs.
   This uses every row (GPBag47: 5 slices, GPBag87: 9 slices). ~10–15 min per case.
4. **Cut-rule options** for floating stack nodes (official side, other side, average).

Transistor models, test set (mean of the four per-current medians):

| Case | Official GP 7k | GP_cap | GPBag47 | **GPBag87** |
|---|---|---|---|---|
| NMOS ON | 0.49 % | 0.48 % | 0.39 % | **0.40 %** |
| NMOS OFF | 0.38 % | 0.34 % | 0.30 % | **0.27 %** |
| PMOS ON | 0.38 % | 0.38 % | 0.35 % | **0.33 %** |
| PMOS OFF | 6.08 % | 5.57 % | 4.55 % | **4.29 %** |

Findings:

- GPBag87 was the best model in 17 of the 18 gate cases (table in section 6) and the best
  for the parity checker (97.11 %).
- The cut rule changes the results by < 0.1 % → the official rule is kept.
- One case got worse: **NOR2 11 fell to 74.4 %**, so the overall gate accuracy (97.37 %)
  stayed below GP_cap (97.68 %). The validation corners picked GP_cap for that reason.

## 4. Root cause of the NOR2 11 error

1. **Swap test** (`exp_acc/nor2_diag.py`): GPBag87 everywhere, replacing one case model at a
   time. Only the NMOS ON model changes the NOR2 11 accuracy; the other three change nothing.
   The worst corners are all cold (−30 to −40 °C) with the output at 0 V (errors up to +378 %).
2. **Direct single-transistor SPICE check** at that operating point (NMOS ON, Vd = Vs = 0,
   gate at VDD, the 200 corners' PVT values; `exp_acc/nmos_on_vds0.py`): the GP's gate
   current error was 18–22 % median and **~100 % median below 0 °C** (worst −480 %).
3. **Why**: in NOR2 11 the PMOS stack barely leaks when cold, so the gate current of the two
   ON NMOS is almost the whole leakage. That point is rare in the training data (only 31 of
   10,000 rows have Vd and Vs < 50 mV, 5 of them cold), and all our voltage features are
   divided by kT/q — right for sub-threshold current, but gate tunnelling depends on the
   oxide voltage, not on temperature, so cold corners fall outside the trained feature range.

## 5. Round 2 — fixes for the remaining weak spots (`exp_acc/`, started 9 Nov 19:45)

### Step 1: plain-voltage features for the ON cases (done)

`FEATS_X` = official features + plain Vgs, Vgd, Vds (not divided by kT/q); NMOS ON and PMOS ON
retrained as GPBag87X on the 87k pool (~14 min each).

Direct SPICE check at the gates' ON operating point (Vd = Vs = rail), gate current error:

| | GPBag87 | **GPBag87X** |
|---|---|---|
| NMOS ON, median | 21.5 % | **4.4 %** |
| NMOS ON, cold corners (< 0 °C), median | 102.9 % | **7.3 %** |
| NMOS ON, worst | 481 % | **32 %** |
| PMOS ON, median / worst | 0.77 % / 40 % | **0.60 % / 6.8 %** |

Transistor validation, median error per current (id / ig / is / ib):

| Case | Official GP 7k | GPBag87 | **GPBag87X** |
|---|---|---|---|
| NMOS ON | 0.52 / 0.37 / 0.50 / 0.49 % | 0.42 / 0.25 / 0.42 / 0.37 % | **0.36 / 0.16 / 0.34 / 0.37 %** |
| PMOS ON | 0.54 / 0.10 / 0.52 / 0.42 % | 0.44 / 0.08 / 0.44 / 0.36 % | **0.33 / 0.05 / 0.33 / 0.36 %** |

Result with GPBag87 for the OFF cases and GPBag87X for the ON cases ("GPX"):
**gates 98.65 %** (validation corners 98.61 %), median 0.67 %, p95 4.90 %; **NOR2 11: 74.4 % → 97.0 %**;
parity 97.11 % (unchanged: the parity checker only uses NAND2).

### Step 2: MLP ensembles and GP + MLP averaging (done, 9 Nov 20:13 – 10 Nov 01:40)

Per case: 3 seeds × (ridge fit + one MLP per current on its residual), 3 × 128 tanh,
up to 4000 epochs, patience 200, FEATS_X, 87k rows; 4 cases in parallel (1 core each,
5–45 min per network).

Transistor models, validation / test (mean of the four per-current medians):

| Case | GPBag87 | GPBag87X | MLP87 | GPX+MLP average |
|---|---|---|---|---|
| NMOS OFF | 0.261 / 0.269 % | (same as GPBag87) | 0.378 / 0.382 % | **0.254 / 0.268 %** |
| PMOS ON | 0.329 / 0.334 % | **0.267 / 0.282 %** | 0.704 / 0.728 % | 0.402 / 0.411 % |
| PMOS OFF | **3.94 / 4.29 %** | (same as GPBag87) | 6.44 / 6.14 % | 5.10 / 5.02 % |

(NMOS ON: GPBag87X best as well, 0.31 % validation.) The MLPs are less accurate than the
GPs at transistor level in every case.

Gates and parity (official corners; validation-corner gate accuracy in brackets):

| Model set | Gates | median / p95 | NOR2 11 | Parity |
|---|---|---|---|---|
| GPBag87 | 97.37 % (97.50) | 0.71 / 6.19 % | 74.41 % | 97.11 % |
| **GPX** (GP, plain-voltage features for ON) | **98.65 %** (98.61) | 0.67 / 4.90 % | 96.99 % | 97.11 % |
| MLP87 | 97.52 % (97.19) | 1.66 / 7.51 % | 97.69 % | 96.67 % |
| GPX+MLP average, all cases | 98.34 % (98.19) | 0.99 / 5.43 % | 97.68 % | 97.12 % |
| GPX, MLP for the ON cases | **98.72 %** (98.67) | 0.63 / 4.72 % | 98.26 % | 97.11 % |
| GPX, GP/MLP average for the ON cases | 98.68 % (98.64) | 0.66 / 4.74 % | 97.64 % | 97.11 % |

The MLP helps only NOR2 11 (ON-NMOS gate current at Vds = 0); everywhere else the GP is
better. "GPX, MLP for ON" is best on the validation corners by 0.06 %, while GPX is best
on the transistor validation split for every case. Because the project document reserves
gate SPICE results for evaluation, the final choice is made on the transistor validation
split → **GPX** (all four models are GPs); the 0.07 % difference at gate level is within
the corner-to-corner noise (see the fresh check below).

## 5b. Fresh independent check (10 Nov 01:41 – 02:28)

Data never used anywhere before: 2,000 new transistor samples per case (seeds 9000–9003),
100 new corners (seed 9999), 1,800 gate and 25,600 parity SPICE runs (0 failures, 0 logic
errors). Every model set scored once; nothing tuned or chosen here.

| Model set | Gates | median / p95 | Parity | parity median / bias |
|---|---|---|---|---|
| Official GP 7k | 95.13 % | 2.20 / 12.18 % | 95.08 % | 4.12 / −3.77 % |
| GPBag87 | 97.03 % | 0.74 / 7.35 % | 97.14 % | 2.11 / −1.86 % |
| **GPX** | **98.61 %** | 0.71 / 5.65 % | **97.14 %** | 2.14 / −1.92 % |
| MLP87 | 97.38 % | 1.54 / 8.84 % | 96.38 % | 2.90 / −1.49 % |
| GPX+MLP average | 98.23 % | 0.97 / 6.62 % | 97.05 % | 2.44 / −1.86 % |
| GPX, MLP for ON | 98.67 % | 0.69 / 5.42 % | 97.14 % | 2.13 / −1.92 % |
| GPX, average for ON | 98.64 % | 0.71 / 5.58 % | 97.14 % | 2.14 / −1.92 % |

Composition on the fresh corners (sum of SPICE NAND2 vs SPICE circuit): bias −1.79 %,
accuracy 97.71 % (official corners: −2.15 %). The GPX results reproduce on unseen data
(gates 98.65 → 98.61 %, parity 97.11 → 97.14 %), so they are not tuned to the official corners.

## 5c. Dataset-size study (10 Nov 02:29 – 05:05)

GPX model type (averaged GP; plain-voltage features for ON) trained on increasing numbers
of rows (2k–7k: first rows of the original training split; 27k/47k/87k: + our extra blocks).

| Training rows per case | NMOS ON | NMOS OFF | PMOS ON | PMOS OFF | Gates (val corners) | Gates (official) | Parity |
|---|---|---|---|---|---|---|---|
| 2,000 | 0.67 % | 0.76 % | 0.52 % | 6.94 % | 92.00 % | 92.04 % | 90.29 % |
| 4,000 | 0.51 % | 0.49 % | 0.38 % | 6.81 % | 93.85 % | 94.32 % | 92.71 % |
| 7,000 | 0.41 % | 0.36 % | 0.33 % | 5.68 % | 96.85 % | 96.87 % | 95.71 % |
| 27,000 | 0.32 % | 0.27 % | 0.28 % | 5.12 % | 97.78 % | 97.66 % | 96.72 % |
| 47,000 | 0.30 % | 0.29 % | 0.27 % | 3.97 % | 96.95 % | 96.96 % | 95.89 % |
| 87,000 | 0.31 % | 0.26 % | 0.27 % | 3.94 % | **98.61 %** | **98.65 %** | **97.11 %** |

(Transistor columns: validation mean of the four per-current medians.)

- Transistor accuracy improves steadily up to ~27k rows and then flattens (except PMOS OFF).
- With the same 7,000 rows as the official model, the plain-voltage features alone raise the
  gates from 95.58 % to 96.87 %.
- Gate accuracy is not monotonic: the 47k OFF models (GPBag47) put back the systematic bias at
  the "PMOS OFF holding the full VDD" point (INV 1, NAND2 11, AND2 00: ~94.8 %). That single
  operating point is sparsely covered, so which rows fall near it matters; with 87k rows it is
  covered well enough at every slice. 87k is the smallest tested size that reaches ~98.6 %.

---

## 6. Case by case: gate accuracy at every stage (official 200 corners)

| Gate | Official | Exp. 1 GP_cap | Round 1 GPBag87 | **Round 2 GPX** | Bias (GPX) |
|---|---|---|---|---|---|
| INV 0 | 96.31 | 97.87 | 99.05 | **99.05** | −0.59 % |
| INV 1 | 95.84 | 97.87 | 98.52 | **98.52** | +0.07 % |
| NAND2 00 | 99.03 | 99.10 | 99.48 | **99.69** | +0.19 % |
| NAND2 01 | 98.57 | 99.09 | 99.45 | **99.48** | −0.09 % |
| NAND2 10 | 99.22 | 99.55 | 99.68 | **99.69** | +0.22 % |
| NAND2 11 | 95.89 | 97.90 | 98.54 | **98.55** | +0.07 % |
| NOR2 00 | 96.32 | 97.88 | 99.05 | **99.05** | −0.59 % |
| NOR2 01 | 96.70 | 98.56 | 98.79 | **98.79** | +0.07 % |
| NOR2 10 | 98.78 | 99.40 | 99.57 | **99.72** | +0.02 % |
| NOR2 11 | 73.32 | 89.15 | 74.41 | **96.99** | +0.89 % |
| AND2 00 | 95.85 | 97.87 | 98.52 | **98.53** | +0.06 % |
| AND2 01 | 95.78 | 97.22 | 97.49 | **97.49** | −1.79 % |
| AND2 10 | 95.97 | 97.89 | 98.39 | **98.40** | −0.35 % |
| AND2 11 | 95.86 | 97.55 | 97.98 | **97.99** | −1.21 % |
| OR2 00 | 95.96 | 97.19 | 97.58 | **97.58** | −1.74 % |
| OR2 01 | 97.02 | 97.89 | 98.09 | **98.09** | −1.18 % |
| OR2 10 | 97.81 | 98.36 | 98.97 | **98.96** | −0.92 % |
| OR2 11 | 96.28 | 97.93 | 99.04 | **99.07** | −0.58 % |
| **All (accuracy)** | **95.58** | **97.68** | **97.37** | **98.65** | −0.23 % |
| median error | 2.16 % | 1.26 % | 0.71 % | **0.67 %** | |
| p95 error | 10.34 % | 5.73 % | 6.19 % | **4.90 %** | |
| cases ≥ 99 % | 2 | 4 | 7 | **7** | |

What improved which cases:

- **More data + averaged GPs** fixed the "OFF transistor holding the full VDD" cases
  (INV, NAND2 11, NOR2 00, AND2 00/10, OR2 11): systematic bias −2.4 % / +1.1 % → within ±0.6 %.
- **Plain-voltage features** fixed NOR2 11 (ON-NMOS gate current at Vds = 0, cold corners).
- **Still below 98.5 %**: the two-stage gates AND2 01/11 and OR2 00/01 (bias −1.2 to −1.8 %)
  and NOR2 11 (97.0 %). For the two-stage gates the cause is the internal node between the
  stages: it is not exactly at 0 / VDD, but the transistor models only accept a gate
  voltage of exactly 0 or VDD (that is how the training data is defined). This is the same
  effect as the parity composition error below.

## 7. Parity checker at every stage (official 200 corners)

| | Official | GP_cap | GPBag47 | GPBag87 | **GPX** |
|---|---|---|---|---|---|
| accuracy | 95.33 % | 96.86 % | 95.86 % | 97.11 % | **97.11 %** |
| median error | 4.08 % | 2.70 % | 3.31 % | 2.33 % | **2.35 %** |
| p95 error | 11.24 % | 7.89 % | 10.38 % | 7.55 % | **7.39 %** |
| bias | −3.67 % | −2.28 % | −1.97 % | −2.10 % | **−2.12 %** |

**Limit of the prescribed method**: summing the stand-alone *SPICE* NAND2 leakages already
differs from the full-circuit SPICE by −2.15 % (median), i.e. ~97.4 % accuracy even with
perfect gate models. The ML bias (−2.12 %) now equals this composition error, so the ML part
of the parity error is almost zero. Going further needs gate models that use the real
(slightly off-rail) input voltages — planned as an additional, clearly labelled result.

## 8. Why 16nm HP is hard (our own simulation, `scratch_cards.py`)

Single NMOS / PMOS at VDD = 0.9 V, nominal parameters, L = node, leakage per µm width,
for every model card in `models/models`:

| Card | Leakage 25 °C (nA/µm) | 125 °C | Gate-current share |
|---|---|---|---|
| **16nm_HP (ours)** | **6,850** | **14,100** | **0.07 %** |
| 22nm_MGK | 2,670 | 7,450 | 56 % |
| 32nm_MGK | 1,530 | 4,480 | 57 % |
| 45nm_MGK | 906 | 2,800 | 55 % |
| 22nm_HP | 548 | 1,810 | 0.2 % |
| 32nm_HP | 69 | 341 | 1.5 % |
| 45nm_HP | 18 | 110 | 9 % |
| 16nm_LP | 7.3 | 9.4 | 99 % |
| 22nm_LP | 1.1 | 1.9 | 97 % |
| 32nm_LP | 0.27 | 1.2 | 84 % |
| 45nm_LP | 0.06 | 0.7 | 53 % |

16nm_HP has the highest leakage of all cards (sub-threshold dominated), which explains
(a) the large composition error in the circuit and (b) the noise floor in the OFF cases:
with almost no gate current, a pinched-off channel leaves only ~1e-14 A, where SPICE itself
is noisy. 68 % of the PMOS OFF current values are below 1e-13 A; on the currents that matter
for gates (≥ 1e-11 A) the PMOS OFF error is small:

| PMOS OFF test error by current size | Official GP 7k | GP_cap |
|---|---|---|
| ≥ 1e-11 A (gate-relevant), median / p95 | 0.88 % / 7.3 % | 0.62 % / 6.8 % |
| 1e-13 … 1e-11 A, median | 5.1 % | 4.1 % |
| < 1e-13 A (noise floor), median | 5.2 % | 4.8 % |

(GPBag87: 0.26 % / 0.27 % median for id / is on currents ≥ 1e-11 A, validation set.)

## 9. Next steps

1. ~~Round 2 (MLP and GP + MLP averaging)~~ done → GPX chosen on the transistor validation split.
2. ~~Fresh independent check~~ done → results reproduce (gates 98.61 %, parity 97.14 %).
3. ~~Dataset-size study~~ done.
4. Move GPX into the main pipeline (after approval), regenerate all deliverables and figures,
   update README / PROJECT_GUIDE.
5. Optional: gate-voltage-aware models (training rows with the gate slightly off the rail) to
   remove the two-stage gate bias and the parity composition error; report only as an extra
   result next to the prescribed one.
6. Write the report (deliverable 10).

## Files

| Path | What |
|---|---|
| `exp_extra/gen_extra.py`, `run_exp.py` | Experiment 1 (40k extra rows, GP_cap, MLP ensemble) |
| `exp_acc/gen_data.py` | 40k more rows per case → `exp_acc/data/*_more_*.csv` |
| `exp_acc/gate_val_spice.py` | 100 validation corners + gate SPICE → `exp_acc/out/gate_spice_val.csv` |
| `exp_acc/common.py`, `train.py`, `evaluate.py`, `run.sh` | Round 1 (GPBag47/87, cut rules) → `exp_acc/out/summary.csv` |
| `exp_acc/nor2_diag.py`, `nmos_on_vds0.py`, `percase.py` | NOR2 11 diagnosis, per-case tables |
| `exp_acc/common2.py`, `step1_gpx.py`, `step2_mlp.py`, `evaluate2.py`, `run2.sh` | Round 2 → `exp_acc/out/round2_*.csv`, `run2.log` |
| `exp_acc/quick_gpx.py`, `val_table.py` | Interim GPX evaluation, per-current validation table |
| `exp_acc/common3.py`, `fresh_check.py`, `size_study.py`, `overnight.sh` | Fresh check → `exp_acc/fresh/`; size study → `exp_acc/out/size_study.csv`; log `overnight.log` |
| `scratch_datacheck*.py`, `scratch_cards.py` | Data verification, model-card leakage comparison |
