"""Fit GREPS NIfTI images and write parameter maps."""

from __future__ import annotations

import os
import time

import nibabel as nib
import numpy as np

from .fit import fit

__all__ = ["load_data", "fit_nifti", "save_map"]


def load_data(magnitude, phase, mask=None, b1=None, flip_angle=20.0, b1_units="percent"):
    """Assemble the fit input from NIfTI files.

    magnitude, phase: 4D images (one volume per phase increment), phase in radians.
    mask:  3D image, voxels > 0.5 are fitted (default: all voxels).
    b1:    3D map of actual / nominal flip angle, in percent (100 = nominal, default, as MagPhsT2)
           or as a fraction (1 = nominal) with b1_units="fraction".

    Returns ``data`` (n_voxels, n_increments) complex, flip angles per voxel, the boolean
    mask and the affine of the magnitude image.
    """
    if b1_units not in ("fraction", "percent"):
        raise ValueError("b1_units must be 'fraction' or 'percent'")
    mag_img = nib.load(magnitude)
    shape3 = mag_img.shape[:3]
    m = np.ones(shape3, bool) if mask is None else nib.load(mask).get_fdata() > 0.5
    mag = np.asanyarray(mag_img.dataobj)[m].astype(np.float64)
    phs = np.asanyarray(nib.load(phase).dataobj)[m].astype(np.float64)
    if mag.ndim != 2 or mag.shape != phs.shape:
        raise ValueError(f"magnitude {mag.shape} and phase {phs.shape} must be 4D with matching shape")
    if b1 is None:
        fa = np.full(m.sum(), float(flip_angle))
    else:
        scale = 100.0 if b1_units == "percent" else 1.0
        fa = flip_angle * nib.load(b1).get_fdata()[m] / scale
    return mag * np.exp(1j * phs), fa, m, mag_img.affine


def save_map(values, mask, affine, path, dtype=np.float64):
    vol = np.zeros(mask.shape, dtype)
    vol[mask] = values
    nib.save(nib.Nifti1Image(vol, affine), path)
    return os.path.abspath(path)


def fit_nifti(
    magnitude,
    phase,
    tr,
    flip_angle,
    phase_increments,
    output_dir,
    mask=None,
    b1=None,
    b1_units="percent",
    prefix="",
    diagnostics=False,
    verbose=True,
    **fit_kwargs,
):
    """Fit NIfTI images and write ``<prefix>T1.nii.gz``, ``T2``, ``amplitude`` (T1/T2 in ms) and
    ``no_bound_mask`` (1 where no parameter ended on a bound).

    With ``diagnostics=True`` also writes ``cost``, ``n_iter`` and ``status`` maps.
    Further keyword arguments go to :func:`t2greps.fit`. Returns (paths, FitResult).
    """
    t0 = time.perf_counter()
    data, fa, m, affine = load_data(magnitude, phase, mask, b1, flip_angle, b1_units)
    t1 = time.perf_counter()
    res = fit(data, fa, tr, phase_increments, **fit_kwargs)
    t2 = time.perf_counter()
    if verbose:
        print(f"t2greps: fitted {m.sum():,} voxels in {t2 - t1:.1f} s (loading {t1 - t0:.1f} s)")
        print(f"t2greps: {res.summary()}")
    os.makedirs(output_dir, exist_ok=True)
    maps = {"T1": res.T1, "T2": res.T2, "amplitude": res.amplitude}
    paths = {"no_bound_mask": save_map(res.no_bound, m, affine, os.path.join(output_dir, f"{prefix}no_bound_mask.nii.gz"),
                                       dtype=np.uint8)}
    if diagnostics:
        maps.update(cost=res.cost, n_iter=res.n_iter, status=res.status)
    paths.update({k: save_map(v, m, affine, os.path.join(output_dir, f"{prefix}{k}.nii.gz")) for k, v in maps.items()})
    return paths, res
