"""Signal model vs the C++ EPG of MagPhsT2 (stored values) and analytic limits."""

from pathlib import Path

import numpy as np
import pytest

import t2greps

REF = np.load(Path(__file__).parent / "data" / "pyepg_reference.npz")
ARGS = [REF[k] for k in ("T1", "T2", "flip_angle", "phase_increment", "tr")]


def measured(s_cpp):
    """C++/equation_grep convention s -> measured convention (-i s)."""
    return -1j * s_cpp


def test_steady_state_matches_converged_cpp():
    s_ref, n = measured(REF["s_tol1e-13"]), REF["n_tol1e-13"]
    ok = n > 0
    assert ok.mean() > 0.95
    s = t2greps.signal(*ARGS)
    # Limited by how far the C++ stepping (tol 1e-13 per TR) is from the true steady state.
    assert np.max(np.abs(s[ok] - s_ref[ok])) < 1e-6
    assert np.median(np.abs(s[ok] - s_ref[ok])) < 1e-9


@pytest.mark.parametrize("tol", ["1e-6", "1e-8"])
def test_stepped_reproduces_cpp_exactly(tol):
    s_ref, n_ref = measured(REF[f"s_tol{tol}"]), REF[f"n_tol{tol}"]
    for i in range(0, len(n_ref), 3):
        s, n = t2greps.signal_stepped(*(a[i] for a in ARGS), tol=float(tol))
        assert n == n_ref[i]
        if n >= 0:
            assert abs(s - s_ref[i]) < 1e-12


def test_ernst_limit():
    # T2 -> 0: no coherences survive, FLASH steady state with phase -pi/2 removed.
    T1, fa, TR = 1000.0, 20.0, 10.0
    e1 = np.exp(-TR / T1)
    ernst = np.sin(np.deg2rad(fa)) * (1 - e1) / (1 - np.cos(np.deg2rad(fa)) * e1)
    assert abs(t2greps.signal(T1, 1e-3, fa, 2.0, TR) - ernst) < 1e-12


def test_broadcasting_and_scalar():
    s = t2greps.signal(1000.0, [40.0, 80.0], 20.0, [[1.0], [2.0], [4.0]], 10.8)
    assert s.shape == (3, 2)
    assert s[1, 0] == t2greps.signal(1000.0, 40.0, 20.0, 2.0, 10.8)
    assert isinstance(t2greps.signal(1000.0, 80.0, 20.0, 2.0, 10.8), complex)


def test_adaptive_orders_vs_many_orders():
    """Default truncation (from T2/TR) vs 8192 orders over a wide parameter range."""
    rng = np.random.default_rng(7)
    n = 400
    T2 = np.exp(rng.uniform(np.log(5), np.log(1000), n))
    T1 = np.minimum(T2 * np.exp(rng.uniform(0, np.log(40), n)), 10000)
    fa = rng.uniform(3, 40, n)
    q = rng.choice([0.5, 1, 2, 3, 5, 10, 20], n)
    TR = rng.choice([5.0, 7.0, 10.8, 13.7, 20.0], n)
    a = t2greps.signal(T1, T2, fa, q, TR)
    b = t2greps.signal(T1, T2, fa, q, TR, n_states=8192)
    assert np.max(np.abs(a - b) / np.abs(b)) < 5e-8


def test_measured_convention_matches_equation_grep():
    """equation_grep compared data with |s| exp(-i (angle(s) - pi/2)), i.e. conj of our signal."""
    i = 0
    s = t2greps.signal(*(a[i] for a in ARGS))
    s_cpp = REF["s_tol1e-13"][i]
    assert abs(abs(s) - abs(s_cpp)) < 2e-6
    assert abs(np.exp(1j * np.angle(s)) - np.exp(1j * (np.angle(s_cpp) - np.pi / 2))) < 1e-4
