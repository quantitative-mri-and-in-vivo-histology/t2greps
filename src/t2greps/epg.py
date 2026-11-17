"""Steady-state signal of gradient echo with partial (quadratic) RF spoiling (GREPS).

The signal is the steady state of the extended phase graph (EPG) of a GRE
sequence whose RF phase follows phi_n = Q n(n+1)/2. In a frame that rotates
with this phase, every TR applies the same linear map, so the steady state is
the solution of one linear system. That system couples neighbouring dephasing
orders only and reduces to a scalar complex continued fraction over the orders
plus a 3x3 solve for order 0. No stepping and no convergence tolerance are
involved. Order k has spent at least k TRs as transverse magnetisation, which
motivates a relaxation-based truncation rule: max(16, ceil(16 T2 / TR)) orders are
kept per evaluation. This is a heuristic, not a bound. Against 16384 orders the
relative error was below 2e-15 for flip angles 3-90 deg, increments 0.5-20 deg,
TR 3-50 ms and T2 5-1000 ms.

:func:`signal_stepped` is a port of ``EPG::StepsToSS`` from MagPhsT2/EPGpp: it
steps TR by TR from equilibrium and stops when the magnitude changes by less
than ``tol``. It reproduces what MagPhsT2 computed (including its bias at loose
tolerances) and serves as an independent check.

Conventions: times in ms, angles in degrees, M0 = 1. The complex signal is
returned as measured (``abs`` = magnitude, ``angle`` = phase of the
background-corrected GREPS images), i.e. what MagPhsT2 compared its data with.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

__all__ = ["signal", "signal_stepped", "phase_table", "n_orders"]

ORDERS_PER_T2 = 16.0  # orders kept per T2/TR (heuristic); measured relative error < 2e-15
MIN_ORDERS = 16


@njit(inline="always")
def n_orders(T2, TR):
    """Number of dephasing orders kept for a given T2 and TR (see module docstring)."""
    return max(MIN_ORDERS, int(np.ceil(ORDERS_PER_T2 * T2 / TR)))

# --------------------------------------------------------------------------------------
# Dual numbers (value, d/dE1, d/dE2) as tuples; numba inlines them into registers.
# --------------------------------------------------------------------------------------


@njit(inline="always")
def _add(x, y):
    return (x[0] + y[0], x[1] + y[1], x[2] + y[2])


@njit(inline="always")
def _sub(x, y):
    return (x[0] - y[0], x[1] - y[1], x[2] - y[2])


@njit(inline="always")
def _mul(x, y):
    return (x[0] * y[0], x[1] * y[0] + x[0] * y[1], x[2] * y[0] + x[0] * y[2])


@njit(inline="always")
def _div(x, y):
    v = x[0] / y[0]
    return (v, (x[1] - v * y[1]) / y[0], (x[2] - v * y[2]) / y[0])


@njit(inline="always")
def _scale(x, s):
    return (x[0] * s, x[1] * s, x[2] * s)


@njit(inline="always")
def _shift(x, s):
    return (x[0] + s, x[1], x[2])


@njit(inline="always")
def _conj(x):
    return (np.conj(x[0]), np.conj(x[1]), np.conj(x[2]))


@njit(inline="always")
def _real(x):
    return (x[0].real, x[1].real, x[2].real)


@njit(inline="always")
def _imag(x):
    return (x[0].imag, x[1].imag, x[2].imag)


@njit(inline="always")
def flip_coefficients(flip_angle):
    """cos^2(a/2), sin^2(a/2), sin(a), cos(a) of the flip angle in degrees."""
    f = np.deg2rad(flip_angle)
    return np.cos(f / 2) ** 2, np.sin(f / 2) ** 2, np.sin(f), np.cos(f)


@njit(inline="always")
def signal_dual(e1v, e2v, a, b, c, d, w, K):
    """Measured-convention signal and its derivatives w.r.t. E1 = exp(-TR/T1), E2 = exp(-TR/T2).

    Keeps orders 0..K-1. ``w`` holds exp(i j Q) for j = 0..>=K (see :func:`phase_table`).
    Returns a dual number (value, d/dE1, d/dE2).
    """
    e1 = (e1v, 1.0, 0.0)
    h = c * c / 2.0
    z0 = 0.0 + 0.0j
    # Orders k >= 1 (co-rotating frame, post-pulse): F_k, G_k = conj(F_-k), Z_k. Above order
    # j the chain is driven only through F_{j-1}, so G_j = alpha_j F_{j-1}; sweep down from 0.
    alpha = (z0, z0, z0)
    for j in range(K - 1, 0, -1):
        wj = w[j]
        zeta = _div((e1v * wj, wj, z0), (1.0 - d * e1v * wj, -d * wj, z0))
        A = _shift(_scale(zeta, -h), a)
        B = _shift(_scale(zeta, h), b)
        beta = _mul((e2v * w[j + 1], z0, w[j + 1]), alpha)
        N = _add(B, _mul(beta, _sub(_mul(A, A), _mul(B, B))))
        D = _shift(_scale(_mul(beta, B), -1.0), 1.0)
        alpha = _mul((e2v * w[j - 1], z0, w[j - 1]), _div(N, D))
    # Order 0: F0_pre = gamma conj(F0_post), Z0_pre = E1 Z0_post + (1 - E1); 3x3 real solve.
    gamma = _mul((e2v * np.conj(w[1]), z0, np.conj(w[1])), _conj(alpha))
    g1, g2 = _real(gamma), _imag(gamma)
    m = (1.0 - e1v, -1.0, 0.0)
    a11 = _shift(_scale(g1, -(a + b)), 1.0)
    a12 = _scale(g2, -(a + b))
    a21 = _scale(g2, b - a)
    a22 = _shift(_scale(g1, a - b), 1.0)
    a23 = _scale(e1, c)
    a31 = _scale(g2, -c)
    a32 = _scale(g1, c)
    a33 = _shift(_scale(e1, -d), 1.0)
    t = _sub(_mul(_scale(m, -c), a33), _mul(a23, _scale(m, d)))
    det = _sub(
        _mul(a11, _sub(_mul(a22, a33), _mul(a23, a32))),
        _mul(a12, _sub(_mul(a21, a33), _mul(a23, a31))),
    )
    p1 = _div(_scale(_mul(a12, t), -1.0), det)  # Re F0_post
    p2 = _div(_mul(a11, t), det)  # Im F0_post
    # EPGpp/MagPhsT2 signal s = -F0_post (demodulated); measured convention is -i s = -p2 + i p1.
    return (complex(-p2[0], p1[0]), complex(-p2[1], p1[1]), complex(-p2[2], p1[2]))


def phase_table(phase_increment, n_max):
    """exp(i j Q) for j = 0..n_max; one row per phase increment Q (degrees)."""
    q = np.deg2rad(np.atleast_1d(np.asarray(phase_increment, float)))
    return np.exp(1j * np.outer(q, np.arange(n_max + 1)))


def max_orders(T2_max, tr):
    """Table size needed for T2 up to ``T2_max``."""
    return MIN_ORDERS + int(np.ceil(ORDERS_PER_T2 * T2_max / tr)) + 1


@njit(parallel=True, cache=True)
def _signal_many(T1, T2, fa, qidx, wtab, TR, n_fixed):
    out = np.empty(T1.shape[0], np.complex128)
    for i in prange(T1.shape[0]):
        a, b, c, d = flip_coefficients(fa[i])
        K = n_fixed if n_fixed > 0 else n_orders(T2[i], TR[i])
        out[i] = signal_dual(np.exp(-TR[i] / T1[i]), np.exp(-TR[i] / T2[i]), a, b, c, d, wtab[qidx[i]], K)[0]
    return out


def signal(T1, T2, flip_angle, phase_increment, tr, n_states=None):
    """Steady-state complex GREPS signal per unit amplitude (array inputs broadcast).

    T1, T2, tr in ms; flip_angle, phase_increment in degrees. ``abs`` of the result is the
    magnitude, ``angle`` the phase as in background-corrected GREPS images.
    ``n_states``: fixed number of dephasing orders (default: chosen from T2/TR).
    """
    args = np.broadcast_arrays(*(np.asarray(v, float) for v in (T1, T2, flip_angle, phase_increment, tr)))
    shape = args[0].shape
    T1, T2, fa, q, TR = (np.ascontiguousarray(v.ravel()) for v in args)
    uq, qidx = np.unique(q, return_inverse=True)
    n_max = n_states if n_states else max(max_orders(T2.max(), TR.min()), 2)
    out = _signal_many(T1, T2, fa, qidx.astype(np.int64), phase_table(uq, n_max + 1), TR, int(n_states or 0))
    return out.reshape(shape) if shape else out[0]


# --------------------------------------------------------------------------------------
# Reference behaviour: TR-by-TR stepping until |d|F0|| < tol (port of EPG::StepsToSS)
# --------------------------------------------------------------------------------------


@njit(cache=True)
def _steps_to_ss(T1, T2, fa, qphi, TR, tol, max_steps, K):
    """Lab-frame stepping with orders -K..K; exact as long as the loop stops before K TRs."""
    a, b, c, d = flip_coefficients(fa)
    e1 = np.exp(-TR / T1)
    e2 = np.exp(-TR / T2)
    F = np.zeros(2 * K + 3, np.complex128)  # order k at index k + K + 1
    Z = np.zeros(K + 1, np.complex128)
    Z[0] = 1.0
    o = K + 1
    phi = 0.0
    f_old = -1.0
    for n in range(min(max_steps, K - 1)):
        phi = np.fmod(phi + n * qphi, 360.0)
        pr = np.deg2rad(phi)
        e, f = np.sin(pr), np.cos(pr)
        g, h = np.sin(2 * pr), np.cos(2 * pr)
        hb, gb, ec, fc = h * b, g * b, e * c, f * c
        L = n + 1  # orders 0..n can be populated before pulse n
        for i in range(L):
            P, M, Zi = F[o + i], F[o - i], Z[i]
            fp = complex(a * P.real + hb * M.real + gb * M.imag + ec * Zi.real + fc * Zi.imag,
                         a * P.imag - hb * M.imag + gb * M.real - fc * Zi.real + ec * Zi.imag)
            fm = complex(hb * P.real + gb * P.imag + a * M.real + ec * Zi.real - fc * Zi.imag,
                         gb * P.real - hb * P.imag + a * M.imag - fc * Zi.real - ec * Zi.imag)
            zn = complex((-ec * P.real + fc * P.imag - ec * M.real + fc * M.imag + 2 * d * Zi.real) / 2,
                         (-fc * P.real - ec * P.imag + fc * M.real + ec * M.imag + 2 * d * Zi.imag) / 2)
            if i > 0:
                F[o + i] = fp
            F[o - i] = fm  # for i = 0 the F(-k) formula wins, as in the C++ code
            Z[i] = zn
        f0 = F[o]
        mag = abs(f0)
        done = abs(mag - f_old) < tol
        # Relaxation and dephasing: every order shifts by +1; Z0 recovers.
        for j in range(o + L, o - L - 1, -1):
            F[j + 1] = F[j] * e2
        F[o - L - 1] = 0.0
        for i in range(L):
            Z[i] *= e1
        Z[0] += 1.0 - e1
        if done:
            s = -f0 * np.exp(-1j * pr)
            return complex(s.imag, -s.real), n  # measured convention: -i s
        f_old = mag
    return complex(np.nan, np.nan), -1


def signal_stepped(T1, T2, flip_angle, phase_increment, tr, tol=1e-6, max_steps=99999):
    """Signal as MagPhsT2 computed it: stepped from equilibrium until |d|F0|| < tol.

    Returns ``(signal, n)`` with ``n`` the C++ ``StepsToSS`` return value (-1 if not
    stopped within ``max_steps``). MagPhsT2 used tol = 1e-8 (before Oct 2024) and
    1e-6 (after). Slow; for comparisons, not for fitting.
    """
    K = 1024
    while True:
        s, n = _steps_to_ss(float(T1), float(T2), float(flip_angle), float(phase_increment), float(tr),
                            float(tol), int(max_steps), K)
        if n >= 0 or K > max_steps:
            return s, n
        K *= 4  # not stopped before the order limit: retry with room for more TRs
