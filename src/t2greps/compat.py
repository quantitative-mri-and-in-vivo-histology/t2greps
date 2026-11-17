"""Drop-in replacements for the MagPhsT2 functions used by existing code.

``cal_T2T1AM`` takes the same arguments as ``T2T1AM.cal_T2T1AM`` (TR in ms, FA in
degrees, B1 map in percent, phase increments in degrees) and writes the same
files (``T1_<label>.nii.gz``, ``T2_...``, ``Am_...``; T1/T2 in ms) next to the
magnitude image or into ``outputdir``. The model is the exact steady state
instead of MagPhsT2's TR-by-TR stepping to a tolerance, see the README.
"""

from __future__ import annotations

import os

import numpy as np

from .epg import signal
from .fit import fit
from .io import load_data, save_map

__all__ = ["cal_T2T1AM", "greps_signal_pyepg"]


def cal_T2T1AM(magn_file, phas_file, mask_file, b1_file, TR, FA, phi, label=None, label_B1=None,
               outputdir=None, **fit_kwargs):
    """Same signature, inputs and output files as ``T2T1AM.cal_T2T1AM`` from MagPhsT2."""
    data, fa, mask, affine = load_data(magn_file, phas_file, mask_file, b1_file, FA, b1_units="percent")
    print(f"t2greps: {mask.sum():,} voxels, TR={TR} ms, FA={FA} deg, phase increments {list(phi)}")
    res = fit(data, fa, TR, phi, **fit_kwargs)
    print(f"t2greps: {res.summary()}")
    base = "" if label is None else "_" + label
    if label_B1 is not None:
        base += "_" + label_B1
    outdir = os.path.dirname(magn_file) if outputdir is None else outputdir
    for name, vals in (("T1", res.T1), ("T2", res.T2), ("Am", res.amplitude)):
        save_map(vals, mask, affine, os.path.join(outdir, f"{name}_{base}.nii.gz"))
    return res


def greps_signal_pyepg(T1, T2, alpha, delta_phi, TR, ABS_ERR_TOL=None):
    """Like ``equation_grep.greps_signal_pyepg``: (magnitude, phase) of the GREPS signal.

    ``ABS_ERR_TOL`` is accepted for compatibility and ignored (the steady state is exact).
    """
    s = signal(T1, T2, alpha, delta_phi, TR)
    return np.abs(s), np.angle(s)


def main(argv=None):
    """Command line of MagPhsT2's T2T1AM.py: magn phas mask b1 -tr -fa -phi [-label -label_B1 -outputdir]."""
    import argparse

    p = argparse.ArgumentParser(description="T2T1AM.py-compatible command line (t2greps)")
    for a in ("magn_file", "phas_file", "mask_file", "b1_file"):
        p.add_argument(a)
    p.add_argument("-tr", type=float, required=True)
    p.add_argument("-fa", type=float, required=True)
    p.add_argument("-phi", type=float, nargs="+", required=True)
    p.add_argument("-label", default=None)
    p.add_argument("-label_B1", default=None)
    p.add_argument("-outputdir", default=None)
    a = p.parse_args(argv)
    cal_T2T1AM(a.magn_file, a.phas_file, a.mask_file, a.b1_file, a.tr, a.fa, a.phi, a.label, a.label_B1,
               a.outputdir)


if __name__ == "__main__":
    main()
