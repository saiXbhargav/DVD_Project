"""Gate topologies, corner handling, SPICE netlists for gates / circuits and the
ML gate-leakage evaluator.

Sizing (TA AND2 example: INV n=1, p=2 units; stacked devices 2 units):
  unit width WU = 1.5 * Lmin = 24 nm, so every device (24-96 nm) lies strictly
  inside the trained range W = 16-128 nm.
    INV   : NMOS 1U, PMOS 2U
    NAND2 : NMOS 2U (series), PMOS 2U (parallel)
    NOR2  : NMOS 1U (parallel), PMOS 4U (series)

Leakage definition (SPICE): total current drawn from all sources at logic 1,
    I_leak = -( I(Vdd) + sum over inputs at '1' of I(Vin) )
(ngspice I(V) > 0 means current entering the source's + terminal).

ML leakage (cut-set method): split all nodes into H (VDD rail, inputs at 1,
internal nodes above VDD/2) and L (the rest). By KCL the leakage is the net
current flowing from H-nodes into the transistors. Per transistor:
  * d and s on the same side (ON devices, or OFF devices with Vds ~ 0):
    only the gate / body currents can cross the cut, so the contribution uses
    Ig and Ib only. This avoids the ON-device channel current at Vds ~ mV,
    which is tiny and poorly represented in the training data.
  * d and s on opposite sides (OFF device holding the voltage):
    contribution = -(sum of its terminal currents at H nodes).
This is the rigorous form of "sum of sub-threshold + gate + body leakage".
"""
from itertools import product

import numpy as np
import pandas as pd

import config as C
import spice_utils as S

WU = 1.5 * C.LMIN

# device: (name, type, d, g, s, b, width multiplier)
_INV = [("MP", "p", "out", "a", "vdd", "vdd", 2), ("MN", "n", "out", "a", "vss", "vss", 1)]
_NAND2 = [("MPA", "p", "out", "a", "vdd", "vdd", 2), ("MPB", "p", "out", "b", "vdd", "vdd", 2),
          ("MNA", "n", "out", "a", "x", "vss", 2), ("MNB", "n", "x", "b", "vss", "vss", 2)]
_NOR2 = [("MPA", "p", "y", "a", "vdd", "vdd", 4), ("MPB", "p", "out", "b", "y", "vdd", 4),
         ("MNA", "n", "out", "a", "vss", "vss", 1), ("MNB", "n", "out", "b", "vss", "vss", 1)]


def _rename(devs, old, new, suffix=""):
    return [(n + suffix, t, *[new if x == old else x for x in (d, g, s, b)], m) for n, t, d, g, s, b, m in devs]


def _stage2(src):
    return [("MP2", "p", "out", src, "vdd", "vdd", 2), ("MN2", "n", "out", src, "vss", "vss", 1)]


GATES = {
    "INV":   {"inputs": ["a"], "devices": _INV, "nodes": ["out"]},
    "NAND2": {"inputs": ["a", "b"], "devices": _NAND2, "nodes": ["x", "out"]},
    "NOR2":  {"inputs": ["a", "b"], "devices": _NOR2, "nodes": ["y", "out"]},
    "AND2":  {"inputs": ["a", "b"], "devices": _rename(_NAND2, "out", "nd") + _stage2("nd"), "nodes": ["x", "nd", "out"]},
    "OR2":   {"inputs": ["a", "b"], "devices": _rename(_NOR2, "out", "nr") + _stage2("nr"), "nodes": ["y", "nr", "out"]},
}


def patterns(gate):
    return ["".join(p) for p in product("01", repeat=len(GATES[gate]["inputs"]))]


# ------------------------------------------------------------------ corners
def load_corners():
    return pd.read_csv(C.DATA_DIR / "corners.csv")


def corner_params(row, dev):
    return {p: row[f"{dev}_{p}"] for p in C.PROC_PARAMS}


# ------------------------------------------------------------------ netlists
def header(row, title):
    return [f"* {title}",
            S.model_card(corner_params(row, "n"), corner_params(row, "p")),
            C.SPICE_OPTIONS,
            f".temp {S.fmt(row['temp'])}",
            f"Vdd vdd 0 {S.fmt(row['vdd'])}",
            "Vss vss 0 0"]


def device_lines(devs, prefix=""):
    out = []
    for name, t, d, g, s, b, m in devs:
        model = "nmos" if t == "n" else "pmos"
        out.append(f"{prefix}{name} {d} {g} {s} {b} {model} W={S.fmt(m * WU)} L={S.fmt(C.LMIN)}")
    return out


def gate_netlist(gate, row):
    g = GATES[gate]
    lines = header(row, f"{gate} corner {int(row['corner'])}")
    lines += [f"V{i} {i} 0 0" for i in g["inputs"]]
    lines += device_lines(g["devices"])
    lines += [".op", ".end"]
    return "\n".join(lines) + "\n"


def spice_leakage(res, inputs, pattern):
    """Leakage from probe results: -(I(Vdd) + sum of I(Vin) for inputs at 1)."""
    leak = -res["i(vdd)"].copy()
    for name, bit in zip(inputs, pattern):
        if bit == "1":
            leak = leak - res[f"i(v{name})"]
    return leak


# ------------------------------------------------------------------ 8-bit parity checker
PARITY_INPUTS = [f"d{i}" for i in range(8)]


def parity_structure():
    """NAND2 instances of the 8-bit parity checker (odd parity: p = d0 ^ d1 ^ ... ^ d7).

    7 XOR2 in a balanced tree (4 -> 2 -> 1); each XOR2 = 4 NAND2:
      n1 = NAND(a, b); n2 = NAND(a, n1); n3 = NAND(b, n1); out = NAND(n2, n3)
    Returns a list of (instance, input_a, input_b, output) in topological order.
    """
    nands = []
    xors = [("x01", "d0", "d1"), ("x23", "d2", "d3"), ("x45", "d4", "d5"), ("x67", "d6", "d7"),
            ("x0123", "x01", "x23"), ("x4567", "x45", "x67"), ("p", "x0123", "x4567")]
    for out, a, b in xors:
        n1, n2, n3 = f"{out}_n1", f"{out}_n2", f"{out}_n3"
        nands += [(f"{out}_g1", a, b, n1), (f"{out}_g2", a, n1, n2),
                  (f"{out}_g3", b, n1, n3), (f"{out}_g4", n2, n3, out)]
    return nands


def parity_netlist(row):
    lines = header(row, f"8-bit parity checker (28 NAND2) corner {int(row['corner'])}")
    lines += [f"V{d} {d} 0 0" for d in PARITY_INPUTS]
    for inst, a, b, out in parity_structure():
        devs = [(f"{n}_{inst}", t, *[{"a": a, "b": b, "out": out, "x": f"{inst}_x"}.get(x, x) for x in (dn, g, s, bb)], m)
                for n, t, dn, g, s, bb, m in _NAND2]
        lines += device_lines(devs)
    lines += [".op", ".end"]
    return "\n".join(lines) + "\n"


def parity_logic(vector):
    """Logic simulation. vector: string of 8 bits d0..d7.
    Returns ({node: bool}, [(instance, pattern 'AB')])."""
    val = {d: bit == "1" for d, bit in zip(PARITY_INPUTS, vector)}
    pats = []
    for inst, a, b, out in parity_structure():
        pats.append((inst, f"{int(val[a])}{int(val[b])}"))
        val[out] = not (val[a] and val[b])
    return val, pats


# ------------------------------------------------------------------ ML gate evaluator
def _logic(node, pattern, inputs):
    if node in inputs:
        return pattern[inputs.index(node)] == "1"
    return None


class _Ctx:
    """Node voltages / logic levels of one gate + pattern over all corners."""

    def __init__(self, gate, pattern, corners, node_v):
        self.g, self.pattern, self.corners = GATES[gate], pattern, corners
        self.n = len(corners)
        self.vdd = corners["vdd"].to_numpy(float)
        self.node_v = {k: np.asarray(v, float) for k, v in node_v.items()}
        # logic level of internal nodes is taken from the SPICE voltages and kept fixed
        self.hi_int = {k: v > self.vdd / 2 for k, v in self.node_v.items()}
        # A stack node between two OFF devices floats at an intermediate voltage. Either
        # side of the cut is exact by KCL; put it on the side of its non-rail neighbour,
        # so the device crossing the cut is the one whose source sits on a rail (its
        # channel-dominated regime, where the transistor models are most accurate).
        for k in self.node_v:
            if k == "out":
                continue
            devs = [dv for dv in self.g["devices"] if k in (dv[2], dv[4])]
            if not devs:
                continue
            off = np.ones(self.n, bool)
            for _, t, d, gn, s, b, m in devs:
                gh = self.high(gn)
                off &= ~gh if t == "n" else gh
            for _, t, d, gn, s, b, m in devs:
                other = s if d == k else d
                if other not in ("vdd", "vss"):
                    self.hi_int[k] = np.where(off, self.high(other), self.hi_int[k])

    def volt(self, node):
        if node == "vdd":
            return self.vdd
        if node == "vss":
            return np.zeros(self.n)
        lv = _logic(node, self.pattern, self.g["inputs"])
        if lv is not None:
            return self.vdd if lv else np.zeros(self.n)
        return self.node_v[node]

    def high(self, node):
        if node == "vdd":
            return np.ones(self.n, bool)
        if node == "vss":
            return np.zeros(self.n, bool)
        lv = _logic(node, self.pattern, self.g["inputs"])
        if lv is not None:
            return np.full(self.n, lv)
        return self.hi_int[node]


def _device_currents(ctx, dev, models):
    """ML terminal currents (n x 4: id, ig, is, ib; > 0 = out of device) of one device."""
    name, t, d, gn, s, b, m = dev
    n, vdd, corners = ctx.n, ctx.vdd, ctx.corners
    gate_hi = ctx.high(gn)
    on = gate_hi if t == "n" else ~gate_hi
    I = np.zeros((n, 4))
    for state in (True, False):
        sel = on == state
        if not sel.any():
            continue
        case = f"{'nmos' if t == 'n' else 'pmos'}_{'on' if state else 'off'}"
        _, vg_rule, vb_rule = C.CASES[case]
        cs = corners[sel]
        vsel = vdd[sel]
        df = pd.DataFrame({
            "vdd": vsel, "temp": cs["temp"].to_numpy(float),
            "w": np.full(sel.sum(), m * WU),
            "vd": ctx.volt(d)[sel], "vs": ctx.volt(s)[sel],
            "vg": vsel if vg_rule == "vdd" else np.full(sel.sum(), vg_rule),
            "vb": vsel if vb_rule == "vdd" else np.full(sel.sum(), vb_rule),
        })
        for p in C.PROC_PARAMS:
            df[p] = cs[f"{t}_{p}"].to_numpy(float)
        I[sel] = models[case].predict(df)
    return I


def ml_solve_nodes(gate, pattern, corners, node_v, models, sweeps=3, iters=24):
    """ML internal-node voltages: solve KCL at each internal node (not 'out') with the
    transistor models, by bisection on [0, VDD], Gauss-Seidel over the nodes.
    Starts from the SPICE values; 'out' is kept at its SPICE value."""
    ctx = _Ctx(gate, pattern, corners, node_v)
    nodes = [k for k in ctx.g["nodes"] if k != "out"]
    for _ in range(sweeps if len(nodes) > 1 else 1):
        for node in nodes:
            devs = [dv for dv in ctx.g["devices"] if node in dv[2:6]]
            lo, hi = np.zeros(ctx.n), ctx.vdd.copy()
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                ctx.node_v[node] = mid
                # net current flowing out of the devices into the node
                f = np.zeros(ctx.n)
                for dv in devs:
                    I = _device_currents(ctx, dv, models)
                    for col, term in enumerate(dv[2:6]):
                        if term == node:
                            f += I[:, col]
                # more current into the node -> node voltage must rise
                lo = np.where(f > 0, mid, lo)
                hi = np.where(f > 0, hi, mid)
            ctx.node_v[node] = 0.5 * (lo + hi)
    return {k: ctx.node_v[k] for k in nodes}


def ml_gate_leakage(gate, pattern, corners, node_v, models, return_devices=False):
    """ML leakage of one gate / input pattern at every corner.

    corners : DataFrame of corners (one row each)
    node_v  : dict node -> array of SPICE node voltages (one per corner)
    models  : dict case -> TransistorModel
    """
    ctx = _Ctx(gate, pattern, corners, node_v)
    high = ctx.high
    total = np.zeros(ctx.n)
    per_dev = {}
    for dev in ctx.g["devices"]:
        name, t, d, gn, s, b, m = dev
        I = _device_currents(ctx, dev, models)
        # cut-set contribution (I columns: id, ig, is, ib; > 0 = out of device)
        H = np.column_stack([high(d), high(gn), high(s), high(b)])
        same = H[:, 0] == H[:, 2]
        gb = np.array([False, True, False, True])
        c_same_hi = (I * (~H & gb)).sum(1)          # d,s in H: current leaving into L via g/b
        c_same_lo = -(I * (H & gb)).sum(1)          # d,s in L: current entering from H via g/b
        c_cross = -(I * H).sum(1)                   # OFF device across the cut
        c = np.where(same, np.where(H[:, 0], c_same_hi, c_same_lo), c_cross)
        total += c
        per_dev[name] = c
    return (total, per_dev) if return_devices else total
