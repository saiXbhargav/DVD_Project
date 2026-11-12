# expected_results/: where each reference file comes from

These files are the outputs of the runs that produced the reported numbers. `verify_final.py` compares `results/` against them. Paths are relative to the repository root. The experiment folders are now in `old/`.

| File | Copied from | Produced by |
|---|---|---|
| `summary.csv` | `old/exp_vg/out/vg_summary_gpx.csv` | `old/exp_vg/eval_vg.py` with `MODELSET=GPX` (final models + real gate voltage) |
| `gate_results.csv` | `old/exp_vg/out/vg_gate_results_gpx.csv` | same run |
| `node_voltages.csv` | `old/exp_vg/out/vg_node_voltages_gpx.csv` | same run |
| `gate_results_fixed.csv` | `old/exp_vg/out/vg_fixed_gate_results_gpx.csv` | same run (fixed-size reference) |
| `gate_ml.csv` | `old/exp_vg/out/vg_gate_ml_gpx.csv` | same run (per gate case and corner, 7 significant digits) |
| `vparity_summary.csv` | `old/exp_vparity/out/vparity_varying_summary.csv` | `old/exp_vparity/vparity.py`, `DEPTHS=0,1`, `LINVDS=1` (optional extension) |
| `kcl_check.csv` | `old/results/kcl_check.csv` | `old/02_data_check.py` (same code as step 2) |
| `model_comparison_val.csv` | `old/results/model_comparison_val.csv` | `old/03_train_models.py` (same comparison as step 3a; 4 model types on the 7,000-row training split) |
| `transistor_metrics.csv` | written by `tools/convert_models.py` | the original model pickles (`old/exp_acc/out/models/GPBag87X_<on case>`, `GPBag87_<off case>`) scored with `final_method.transistor_metrics` on the validation and test splits |

Inputs of those runs and of this pipeline (the same files, verified byte for byte when copied):

| Now | Copied from |
|---|---|
| `data/train_extra/<case>_extra_0b.csv` | `old/exp_extra/data/` |
| `data/train_extra/<case>_more_0b.csv` | `old/exp_acc/data/` |
| `data/corners_w.csv` | `old/exp_width/out/corners_w.csv` |
| `results/gate_spice_w.csv`, `circuit_spice_w.csv` | `old/exp_width/out/` |
| `results/gate_spice.csv`, `circuit_spice.csv` | `old/results/` |
| `models_trained/<case>.joblib` | `old/exp_acc/out/models/GPBag87X_nmos_on`, `GPBag87_nmos_off`, `GPBag87X_pmos_on`, `GPBag87_pmos_off` (class references moved to `final_method.py` by `tools/convert_models.py`; identical predictions) |
