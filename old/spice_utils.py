"""Helpers for building netlists from the PTM model card and running ngspice in batch.

Many independent circuits are simulated in one ngspice process: a master control
script `source`s each netlist, runs `op`, prints the requested vectors and removes
the circuit again. Chunks of circuits run in parallel processes.
"""
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import config as C

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


# ------------------------------------------------------------------ model card
def _split_card(text):
    """Return (nmos_block, pmos_block) of the .pm file."""
    i_n = re.search(r"\.model\s+nmos", text, re.I).start()
    i_p = re.search(r"\.model\s+pmos", text, re.I).start()
    return text[i_n:i_p], text[i_p:]


def _get(block, name):
    m = re.search(rf"\b{name}\s*=\s*({_NUM})", block, re.I)
    if m is None:
        raise KeyError(name)
    return float(m.group(1))


def _set(block, name, value):
    new, n = re.subn(rf"(\b{name}\s*=\s*){_NUM}", lambda m: f"{m.group(1)}{value:.10g}", block, count=1, flags=re.I)
    if n != 1:
        raise KeyError(name)
    return new


_CARD = C.MODEL_FILE.read_text()
_NBLK, _PBLK = _split_card(_CARD)

NOMINAL = {
    "n": {p: _get(_NBLK, p) for p in C.PROC_PARAMS},
    "p": {p: _get(_PBLK, p) for p in C.PROC_PARAMS},
}


def model_card(nparams=None, pparams=None):
    """Model card text with the given process parameters substituted (others nominal)."""
    nb, pb = _NBLK, _PBLK
    for k, v in (nparams or {}).items():
        nb = _set(nb, k, v)
    for k, v in (pparams or {}).items():
        pb = _set(pb, k, v)
    return nb + "\n" + pb


def fmt(x):
    return f"{x:.10g}"


# ------------------------------------------------------------------ running ngspice
def _run_chunk(args):
    netlists, probes = args
    with tempfile.TemporaryDirectory(prefix="ngs_") as td:
        td = Path(td)
        lines = ["* master", ".control", "set numdgt=12"]
        for k, net in enumerate(netlists):
            f = td / f"c{k}.cir"
            f.write_text(net)
            lines += [f"source {f}", "op", f"echo @@ {k}",
                      "print " + " ".join(probes), "remcirc"]
        lines += [".endc", ".end"]
        master = td / "master.cir"
        master.write_text("\n".join(lines) + "\n")
        out = subprocess.run([C.NGSPICE, "-b", str(master)], capture_output=True, text=True).stdout
    return _parse(out, len(netlists), probes)


def run_op(netlists, probes, n_jobs=C.N_JOBS, chunk=C.CHUNK):
    """Run an .op on every netlist and return an (N, len(probes)) array (NaN = failed)."""
    chunks = [(netlists[i:i + chunk], probes) for i in range(0, len(netlists), chunk)]
    if n_jobs == 1 or len(chunks) == 1:
        parts = [_run_chunk(c) for c in chunks]
    else:
        with ProcessPoolExecutor(n_jobs) as ex:
            parts = list(ex.map(_run_chunk, chunks))
    return np.vstack(parts)


def _parse(out, n_rows, probes):
    res = np.full((n_rows, len(probes)), np.nan)
    pat = {p.lower(): j for j, p in enumerate(probes)}
    cur = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("@@"):
            cur = int(s.split()[1])
            continue
        if cur is None or "=" not in s:
            continue
        name, _, val = s.partition("=")
        j = pat.get(name.strip().lower())
        if j is not None:
            try:
                res[cur, j] = float(val.split()[0])
            except ValueError:
                pass
    return res


def _run_sweep_chunk(args):
    jobs, probes = args
    with tempfile.TemporaryDirectory(prefix="ngs_") as td:
        td = Path(td)
        lines = ["* master", ".control", "set numdgt=12"]
        row = 0
        for k, (net, alter_sets) in enumerate(jobs):
            f = td / f"c{k}.cir"
            f.write_text(net)
            lines.append(f"source {f}")
            for alters in alter_sets:
                lines += [f"alter {a}" for a in alters]
                lines += ["op", f"echo @@ {row}", "print " + " ".join(probes)]
                row += 1
            lines.append("remcirc")
        lines += [".endc", ".end"]
        master = td / "master.cir"
        master.write_text("\n".join(lines) + "\n")
        out = subprocess.run([C.NGSPICE, "-b", str(master)], capture_output=True, text=True).stdout
    return _parse(out, row, probes)


def run_sweeps(jobs, probes, n_jobs=C.N_JOBS, chunk=8):
    """jobs: list of (netlist, [[alter, ...], ...]). One .op per alter set.

    Each netlist is loaded once and the alter sets (e.g. input patterns) are
    applied in turn. Returns an array with one row per (job, alter set), in order.
    """
    chunks = [(jobs[i:i + chunk], probes) for i in range(0, len(jobs), chunk)]
    with ProcessPoolExecutor(n_jobs) as ex:
        parts = list(ex.map(_run_sweep_chunk, chunks))
    return np.vstack(parts)


def ngspice_version():
    out = subprocess.run([C.NGSPICE, "-v"], capture_output=True, text=True).stdout
    m = re.search(r"ngspice-\S+", out)
    return m.group(0) if m else out.strip()
