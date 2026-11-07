# Project TODO

## Current

- Setup: PTM 16 nm HP, nominal VDD 0.9 V, 8-bit parity checker.
- Four 10,000-sample transistor datasets and four trained GP models are available.
- Gate validation covers 18 input cases at 200 PVT corners: 2.2% median error and 95.6% mean accuracy.
- Parity validation covers 256 input vectors at the same 200 corners: 4.1% median error and 95.3% mean accuracy.
- Tables, netlists, and plots are available; the final report is pending.

Mean accuracy above is defined as 100% minus mean absolute percentage error.

## Aim

Complete all assignment deliverables, demonstrate reproducibility, and improve unseen-data prediction toward the 99% mean accuracy target while keeping the dataset as small as practical.

## TODO

- [ ] Analyze errors against temperature, VDD, and process parameters; identify the worst transistor, gate, input, and corner cases.
- [ ] Investigate PMOS OFF errors and their contribution to gate and circuit bias.
- [ ] Compare prediction error across dataset sizes and justify the final sample count.
- [ ] Review all figures for readable labels, units, and consistent metrics.
- [ ] Confirm gate sizing, leakage accounting, and the accuracy definition with the TA.
- [ ] Make the run script stop on failure and prevent reuse of stale circuit checkpoints.
- [ ] Record dependency versions and complete a clean run of steps 1–8 with fixed seeds.
- [ ] Write the report covering setup, data checks, model comparison, results, error analysis, and limitations.
- [ ] Explain the prescribed use of SPICE node voltages and the circuit assumptions of ideal inputs and ignored inter-gate loading.
- [ ] Align project documentation and verify the final submission against all ten deliverables.
