"""Voxel-wise fit of T1, T2 and amplitude to complex GREPS data.

Model: ``data_j = amplitude * signal(T1, T2, flip_angle, Q_j, TR)`` for the phase
increments Q_j, least squares over real and imaginary parts (as MagPhsT2).

* The amplitude enters linearly and is solved in closed form for given
  (T1, T2), then clipped to its bounds (variable projection).
* (log T1, log T2) are fitted with Levenberg-Marquardt using exact derivatives
  of the model. A parameter on a bound whose gradient points outward is held
  fixed for that step.
* Each voxel iterates until the relative cost change falls below ``ftol`` or the
  step below ``xtol``; voxels run in parallel on all cores. Results do not
  depend on the number of threads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
from numba import njit, prange

from .epg import flip_coefficients, max_orders, n_orders, phase_table, signal_dual

__all__ = ["fit", "FitResult", "Status"]


class Status:
    """Per-voxel termination reason."""

    CONVERGED_COST = 1  # relative cost decrease < ftol
    CONVERGED_STEP = 2  # step < xtol, or every parameter held on a bound
    NO_DECREASE = 3  # damping grew without finding a lower cost
    MAX_ITER = 4  # max_iter reached


@dataclass
class FitResult:
    T1: np.ndarray  # ms
    T2: np.ndarray  # ms
    amplitude: np.ndarray
    cost: np.ndarray  # sum of squared residuals (real and imaginary parts)
    n_iter: np.ndarray
    status: np.ndarray  # see Status
    no_bound: np.ndarray  # True where no parameter ended on one of its bounds

    @property
    def converged(self):
        return self.status <= Status.CONVERGED_STEP

    def summary(self):
        it = self.n_iter
        return (
            f"{len(it)} voxels | iterations median {np.median(it):.0f}, 99th pct {np.percentile(it, 99):.0f}, "
            f"max {it.max()} | converged {np.mean(self.converged):.2%}, "
            f"no decrease {np.mean(self.status == Status.NO_DECREASE):.2%}, "
            f"max_iter {np.mean(self.status == Status.MAX_ITER):.2%} | "
            f"no parameter on a bound {np.mean(self.no_bound):.2%}"
        )


@njit(inline="always")
def _evaluate(t0, t1, a, b, c, d, wtab, TR, m, dm0, dm1):
    """Model for all increments and derivatives w.r.t. (log T1, log T2)."""
    T1 = np.exp(t0)
    T2 = np.exp(t1)
    K = min(n_orders(T2, TR), wtab.shape[1] - 2)
    e1 = np.exp(-TR / T1)
    e2 = np.exp(-TR / T2)
    de1 = e1 * TR / T1  # dE1/dlog T1
    de2 = e2 * TR / T2  # dE2/dlog T2
    for q in range(wtab.shape[0]):
        v, d1, d2 = signal_dual(e1, e2, a, b, c, d, wtab[q], K)
        m[q] = v
        dm0[q] = d1 * de1
        dm1[q] = d2 * de2


@njit(inline="always")
def _project(m, dm0, dm1, y, am_lo, am_hi, r, J0, J1):
    """Closed-form amplitude (clipped), residual r = A m - y, its Jacobian, cost."""
    num = den = dnum0 = dnum1 = dden0 = dden1 = 0.0
    for q in range(m.shape[0]):
        num += (np.conj(m[q]) * y[q]).real
        den += m[q].real ** 2 + m[q].imag ** 2
        dnum0 += (np.conj(dm0[q]) * y[q]).real
        dnum1 += (np.conj(dm1[q]) * y[q]).real
        dden0 += 2.0 * (np.conj(m[q]) * dm0[q]).real
        dden1 += 2.0 * (np.conj(m[q]) * dm1[q]).real
    den = max(den, 1e-300)
    am = num / den
    da0 = (dnum0 * den - num * dden0) / (den * den)
    da1 = (dnum1 * den - num * dden1) / (den * den)
    if am < am_lo:
        am, da0, da1 = am_lo, 0.0, 0.0
    elif am > am_hi:
        am, da0, da1 = am_hi, 0.0, 0.0
    cost = 0.0
    for q in range(m.shape[0]):
        r[q] = am * m[q] - y[q]
        J0[q] = da0 * m[q] + am * dm0[q]
        J1[q] = da1 * m[q] + am * dm1[q]
        cost += r[q].real ** 2 + r[q].imag ** 2
    return am, cost


@njit(cache=True)
def _fit_voxel(y, fa, wtab, TR, start, lo, hi, am_lo, am_hi, max_iter, ftol, xtol):
    nq = wtab.shape[0]
    a, b, c, d = flip_coefficients(fa)
    m = np.empty(nq, np.complex128)
    dm0 = np.empty(nq, np.complex128)
    dm1 = np.empty(nq, np.complex128)
    r = np.empty(nq, np.complex128)
    J0 = np.empty(nq, np.complex128)
    J1 = np.empty(nq, np.complex128)
    rc = np.empty(nq, np.complex128)
    J0c = np.empty(nq, np.complex128)
    J1c = np.empty(nq, np.complex128)

    t0, t1 = start[0], start[1]
    _evaluate(t0, t1, a, b, c, d, wtab, TR, m, dm0, dm1)
    am, cost = _project(m, dm0, dm1, y, am_lo, am_hi, r, J0, J1)
    lam = 1e-2
    status = 4
    it = 0
    while it < max_iter:
        it += 1
        h00 = h01 = h11 = g0 = g1 = 0.0
        for q in range(nq):
            h00 += J0[q].real ** 2 + J0[q].imag ** 2
            h11 += J1[q].real ** 2 + J1[q].imag ** 2
            h01 += (np.conj(J0[q]) * J1[q]).real
            g0 += (np.conj(J0[q]) * r[q]).real
            g1 += (np.conj(J1[q]) * r[q]).real
        free0 = not ((t0 <= lo[0] and g0 > 0.0) or (t0 >= hi[0] and g0 < 0.0))
        free1 = not ((t1 <= lo[1] and g1 > 0.0) or (t1 >= hi[1] and g1 < 0.0))
        A00 = h00 * (1.0 + lam) + 1e-12
        A11 = h11 * (1.0 + lam) + 1e-12
        s0 = s1 = 0.0
        if free0 and free1:
            det = A00 * A11 - h01 * h01
            s0 = -(A11 * g0 - h01 * g1) / det
            s1 = -(A00 * g1 - h01 * g0) / det
        elif free0:
            s0 = -g0 / A00
        elif free1:
            s1 = -g1 / A11
        else:
            status = 2
            break
        c0 = min(max(t0 + s0, lo[0]), hi[0])
        c1 = min(max(t1 + s1, lo[1]), hi[1])
        if abs(c0 - t0) < xtol and abs(c1 - t1) < xtol:
            status = 2
            break
        _evaluate(c0, c1, a, b, c, d, wtab, TR, m, dm0, dm1)
        am_c, cost_c = _project(m, dm0, dm1, y, am_lo, am_hi, rc, J0c, J1c)
        if cost_c < cost:
            rel = (cost - cost_c) / max(cost, 1e-300)
            t0, t1, am, cost = c0, c1, am_c, cost_c
            for q in range(nq):
                r[q], J0[q], J1[q] = rc[q], J0c[q], J1c[q]
            lam = max(lam / 3.0, 1e-12)
            if rel < ftol:
                status = 1
                break
        else:
            lam *= 4.0
            if lam > 1e12:
                status = 3
                break
    return np.exp(t0), np.exp(t1), am, cost, it, status


@njit(parallel=True, cache=True)
def _fit_all(data, fa, wtab, TR, start, lo, hi, am_lo, am_hi, max_iter, ftol, xtol, out, n_iter, status):
    for i in prange(data.shape[0]):
        T1, T2, am, cost, it, st = _fit_voxel(
            data[i], fa[i], wtab, TR, start, lo, hi, am_lo, am_hi, max_iter, ftol, xtol
        )
        out[i, 0], out[i, 1], out[i, 2], out[i, 3] = T1, T2, am, cost
        n_iter[i], status[i] = it, st


def fit(
    data,
    flip_angle,
    tr,
    phase_increments,
    *,
    T1_start=1000.0,
    T2_start=80.0,
    T1_bounds=(1.0, 10000.0),
    T2_bounds=(1.0, 1000.0),
    amplitude_bounds=(1.0, 50000.0),
    max_iter=200,
    ftol=1e-12,
    xtol=1e-10,
    n_threads=None,
):
    """Fit T1, T2 [ms] and amplitude per voxel.

    data:             complex (n_voxels, n_increments), ``magnitude * exp(1j * phase)`` of the
                      background-corrected GREPS images, columns in the order of ``phase_increments``.
    flip_angle:       actual flip angle in degrees, scalar or per voxel (nominal times B1).
    tr:               repetition time in ms.
    phase_increments: RF phase increments in degrees, one per column of ``data``.

    Start values and bounds default to those of MagPhsT2. ``FitResult.no_bound`` marks voxels
    where no parameter ended on a bound; elsewhere at least one parameter is limited by its
    bound (e.g. T2 at its lower bound, where T1 is undetermined), so values are less reliable.
    """
    data = np.ascontiguousarray(np.atleast_2d(data), np.complex128)
    n = data.shape[0]
    if data.shape[1] != len(phase_increments):
        raise ValueError(f"data has {data.shape[1]} columns but {len(phase_increments)} phase increments")
    fa = np.ascontiguousarray(np.broadcast_to(np.asarray(flip_angle, float), (n,)))
    if n_threads:
        numba.set_num_threads(n_threads)
    # Spread slow voxels (CSF, edges) evenly over the threads. Voxels are independent,
    # so the order does not affect any result.
    order = np.random.default_rng(0).permutation(n)
    out = np.empty((n, 4))
    n_iter = np.empty(n, np.int32)
    status = np.empty(n, np.int8)
    _fit_all(
        data[order], fa[order], phase_table(phase_increments, max_orders(T2_bounds[1], tr) + 2), float(tr),
        np.log([T1_start, T2_start]), np.log([T1_bounds[0], T2_bounds[0]]),
        np.log([T1_bounds[1], T2_bounds[1]]), float(amplitude_bounds[0]), float(amplitude_bounds[1]),
        int(max_iter), float(ftol), float(xtol), out, n_iter, status,
    )
    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    out, n_iter, status = out[inv], n_iter[inv], status[inv]
    no_bound = np.ones(n, bool)
    for col, (lo, hi) in ((0, T1_bounds), (1, T2_bounds), (2, amplitude_bounds)):
        no_bound &= (out[:, col] > lo * (1 + 1e-6)) & (out[:, col] < hi * (1 - 1e-6))
    return FitResult(out[:, 0], out[:, 1], out[:, 2], out[:, 3], n_iter, status, no_bound)
