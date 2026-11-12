"""exp_acc overnight jobs: shared helpers (model sets, parity sum)."""
from itertools import product

import joblib
import pandas as pd

from common import OUT, C, G, GPCap  # noqa: F401
from common2 import Blend
from evaluate2 import ON, gates  # noqa: F401

PATS = ["00", "01", "10", "11"]


def _try(name, case):
    f = OUT / "models" / f"{name}_{case}.joblib"
    return joblib.load(f) if f.exists() else None


def build_sets():
    """All model sets whose files exist (official always)."""
    sets = {"official_GP7k": {c: joblib.load(C.MODEL_DIR / f"{c}.joblib") for c in C.CASES}}
    base = {c: _try("GPBag87", c) for c in C.CASES}
    if all(v is not None for v in base.values()):
        sets["GPBag87"] = base
        x = {c: _try("GPBag87X", c) for c in ON}
        if all(v is not None for v in x.values()):
            gpx = {c: x[c] if c in ON else base[c] for c in C.CASES}
            sets["GPX"] = gpx
            mlp = {c: _try("MLP87", c) for c in C.CASES}
            if all(v is not None for v in mlp.values()):
                blend = {c: Blend([gpx[c], mlp[c]]) for c in C.CASES}
                sets["MLP87"] = mlp
                sets["GPX+MLP"] = blend
                sets["GPX|ON:MLP"] = {c: mlp[c] if c in ON else gpx[c] for c in C.CASES}
                sets["GPX|ON:blend"] = {c: blend[c] if c in ON else gpx[c] for c in C.CASES}
    return sets


def vec_counts():
    vectors = ["".join(v) for v in product("01", repeat=8)]
    return pd.DataFrame({v: pd.Series([p for _, p in G.parity_logic(v)[1]]).value_counts()
                         .reindex(PATS, fill_value=0) for v in vectors}).T


def parity_sum(nand, cnt, psp):
    """nand: corner x pattern NAND2 leakages -> parity leakage per (corner, vector), merged with SPICE."""
    ml = pd.DataFrame(nand[PATS].to_numpy() @ cnt.to_numpy().T, index=nand.index, columns=cnt.index)
    ml = ml.stack().rename("leak_ml").reset_index()
    ml.columns = ["corner", "vector", "leak_ml"]
    return psp.merge(ml, on=["corner", "vector"])
