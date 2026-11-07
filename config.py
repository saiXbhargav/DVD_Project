"""Project-wide configuration: assignment values, sampling ranges, seeds and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- assignment
MODEL_FILE = ROOT / "models" / "models" / "16nm_HP.pm"
TECH = "16nm_HP"
VDD_NOM = 0.9            # V
LMIN = 16e-9             # m, channel length = feature size
TEST_CIRCUIT = "8-bit parity checker"

# ---------------------------------------------------------------- sampling ranges (spec section 4)
VDD_SPREAD = 0.10                      # VDD ~ U(0.9*VDD_NOM, 1.1*VDD_NOM)
W_RANGE = (1 * LMIN, 8 * LMIN)         # W ~ U(1*Lmin, 8*Lmin)
T_RANGE = (-40.0, 125.0)               # degC
PROC_PARAMS = ["vth0", "toxe", "ndep", "u0", "xj"]
PROC_SPREAD = 0.10                     # each param ~ U(0.9*nom, 1.1*nom)

# ---------------------------------------------------------------- Part A
N_SAMPLES = 10_000
CASES = {
    #  name        device  Vg      Vb      (strings refer to the sampled VDD)
    "nmos_on":  ("n", "vdd", 0.0),
    "nmos_off": ("n", 0.0,   0.0),
    "pmos_on":  ("p", 0.0,   "vdd"),
    "pmos_off": ("p", "vdd", "vdd"),
}
CURRENTS = ["id", "ig", "is", "ib"]
CSV_COLUMNS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb",
               "vth0", "toxe", "ndep", "u0", "xj"] + CURRENTS

SPLIT = (0.70, 0.15, 0.15)             # train / val / test

# ---------------------------------------------------------------- Part B / C
N_CORNERS = 200

# ---------------------------------------------------------------- seeds
SEED_DATA = 2026        # transistor datasets (offset per case)
SEED_SPLIT = 42         # train/val/test split
SEED_MODEL = 7          # ML model training
SEED_CORNERS = 1234     # 200 gate / circuit corners

# ---------------------------------------------------------------- SPICE
NGSPICE = "ngspice"
SPICE_OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"
N_JOBS = 4              # parallel ngspice processes
CHUNK = 250             # circuits per ngspice process

# ---------------------------------------------------------------- folders
DATA_DIR = ROOT / "data"
NETLIST_DIR = ROOT / "netlists"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
MODEL_DIR = ROOT / "models_trained"
for _d in (DATA_DIR, NETLIST_DIR, RESULTS_DIR, FIG_DIR, MODEL_DIR):
    _d.mkdir(exist_ok=True)
