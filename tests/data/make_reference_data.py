"""Generate pyepg_reference.npz (next to this file) with the C++ EPG of the MagPhsT2 fork.

    python tests/data/make_reference_data.py tests/data/pyepg_reference.npz

Needs ``pyepg`` built from MagPhsT2 (pip install path/to/MagPhsT2). Run once; the
tests then compare against the stored values without any C++ build.

Stored signal convention: as in equation_grep, s = (GetReA + i GetImA) exp(-i phi_n).
"""
import sys

import numpy as np
import pyepg


def cpp(T1, T2, fa, q, TR, tol):
    e = pyepg.PyEPG()
    e.SetParameters(1.0, T1, T2)
    e.Equilibrium()
    n = e.StepsToSS(fa, q, TR, tol)
    return (e.GetReA() + 1j * e.GetImA()) * np.exp(-1j * np.deg2rad(e.GetPhase())), n


rng = np.random.default_rng(42)
n = 300
T2 = np.exp(rng.uniform(np.log(5), np.log(1000), n))
T1 = np.minimum(T2 * np.exp(rng.uniform(0, np.log(40), n)), 10000)
fa = rng.uniform(5, 35, n)
q = rng.choice([1, 1.5, 2, 3, 4, 5, -2, 10], n)
TR = rng.choice([7.0, 9.5, 10.8, 13.7], n)
out = {"T1": T1, "T2": T2, "flip_angle": fa, "phase_increment": q, "tr": TR}
for name, tol in (("tol1e-6", 1e-6), ("tol1e-8", 1e-8), ("tol1e-13", 1e-13)):
    r = [cpp(*v, tol) for v in zip(T1, T2, fa, q, TR, strict=True)]
    out[f"s_{name}"] = np.array([x[0] for x in r])
    out[f"n_{name}"] = np.array([x[1] for x in r])
np.savez_compressed(sys.argv[1], **out)
print("wrote", sys.argv[1], {k: v.shape for k, v in out.items()})
