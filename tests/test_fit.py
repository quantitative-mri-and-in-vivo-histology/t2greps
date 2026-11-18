"""Fit: recovery, derivatives, bounds, threads and NIfTI I/O."""

import nibabel as nib
import numpy as np
import pytest

import t2greps
from t2greps.epg import flip_coefficients, phase_table, signal_dual

PHI = [1.0, 2.0, 4.0, 1.5, 3.0, 5.0]
TR = 10.8


def synthetic(n, snr=None, seed=0):
    rng = np.random.default_rng(seed)
    T2 = rng.uniform(30, 150, n)
    T1 = rng.uniform(600, 3000, n)
    fa = rng.uniform(12, 26, n)
    am = rng.uniform(5000, 30000, n)
    y = am[:, None] * t2greps.signal(T1[:, None], T2[:, None], fa[:, None], np.array(PHI)[None], TR)
    if snr:
        sigma = np.mean(np.abs(y)) / snr
        y = y + sigma * (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape))
    return y, fa, T1, T2, am


def test_derivatives_match_finite_differences():
    w = phase_table([2.0], 512)[0]
    a, b, c, d = flip_coefficients(18.0)
    K = 400
    for e1, e2 in ((0.99, 0.85), (0.995, 0.95), (0.9, 0.5)):
        v, d1, d2 = signal_dual(e1, e2, a, b, c, d, w, K)
        h = 1e-7
        fd1 = (signal_dual(e1 + h, e2, a, b, c, d, w, K)[0] - signal_dual(e1 - h, e2, a, b, c, d, w, K)[0]) / (2 * h)
        fd2 = (signal_dual(e1, e2 + h, a, b, c, d, w, K)[0] - signal_dual(e1, e2 - h, a, b, c, d, w, K)[0]) / (2 * h)
        assert abs(d1 - fd1) < 1e-6 * max(1, abs(d1))
        assert abs(d2 - fd2) < 1e-6 * max(1, abs(d2))


def test_noise_free_recovery():
    y, fa, T1, T2, am = synthetic(3000)
    res = t2greps.fit(y, fa, TR, PHI)
    assert res.converged.mean() > 0.99
    for est, true in ((res.T1, T1), (res.T2, T2), (res.amplitude, am)):
        err = np.abs(est - true) / true
        assert np.median(err) < 1e-8
        assert np.percentile(err, 99) < 1e-5


def test_noisy_fit_is_a_minimum():
    """Perturbing the solution never lowers the cost (local optimality)."""
    y, fa, *_ = synthetic(500, snr=50, seed=1)
    res = t2greps.fit(y, fa, TR, PHI)
    ok = res.converged & (res.T1 > 2) & (res.T1 < 9900) & (res.T2 > 2) & (res.T2 < 990)
    for f1, f2 in ((1.001, 1), (0.999, 1), (1, 1.001), (1, 0.999)):
        m = t2greps.signal(res.T1[:, None] * f1, res.T2[:, None] * f2, fa[:, None], np.array(PHI)[None], TR)
        am = np.clip(np.sum((np.conj(m) * y).real, 1) / np.sum(np.abs(m) ** 2, 1), 1, 50000)
        cost = np.sum(np.abs(am[:, None] * m - y) ** 2, 1)
        assert np.all(cost[ok] >= res.cost[ok] * (1 - 1e-9))


def test_bounds_respected():
    y, fa, *_ = synthetic(300, snr=3, seed=2)
    res = t2greps.fit(y, fa, TR, PHI, T1_bounds=(100, 5000), T2_bounds=(10, 300), amplitude_bounds=(10, 40000))
    assert res.T1.min() >= 100 * (1 - 1e-12) and res.T1.max() <= 5000 * (1 + 1e-12)
    assert res.T2.min() >= 10 * (1 - 1e-12) and res.T2.max() <= 300 * (1 + 1e-12)
    assert res.amplitude.min() >= 10 and res.amplitude.max() <= 40000
    # no_bound is False exactly where a parameter sits on a bound.
    on_bound = np.zeros(len(res.T1), bool)
    for v, (lo, hi) in ((res.T1, (100, 5000)), (res.T2, (10, 300)), (res.amplitude, (10, 40000))):
        on_bound |= np.isclose(v, lo, rtol=1e-6) | np.isclose(v, hi, rtol=1e-6)
    assert np.array_equal(res.no_bound, ~on_bound)
    assert 0 < res.no_bound.mean() < 1  # low SNR: some voxels hit a bound, most do not


def test_invalid_flip_angle_not_fitted():
    """Voxels with flip angle <= 0 (e.g. B1 = 0) are marked invalid; the others are unaffected."""
    y, fa, *_ = synthetic(50, seed=3)
    bad = np.array([3, 17, 40])
    fa_bad = fa.copy()
    fa_bad[bad] = [0.0, -5.0, np.nan]
    ref = t2greps.fit(y, fa, TR, PHI)
    res = t2greps.fit(y, fa_bad, TR, PHI)
    assert np.all(res.status[bad] == t2greps.Status.INVALID)
    for v in (res.T1, res.T2, res.amplitude, res.cost):
        assert np.all(np.isnan(v[bad]))
    assert not res.no_bound[bad].any() and not res.converged[bad].any()
    ok = np.setdiff1d(np.arange(50), bad)
    assert np.array_equal(res.T2[ok], ref.T2[ok]) and np.array_equal(res.T1[ok], ref.T1[ok])


def test_results_independent_of_threads():
    y, fa, *_ = synthetic(2000, snr=30, seed=3)
    a = t2greps.fit(y, fa, TR, PHI, n_threads=1)
    b = t2greps.fit(y, fa, TR, PHI, n_threads=4)
    for k in ("T1", "T2", "amplitude", "cost", "n_iter"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k


@pytest.fixture
def nifti_case(tmp_path):
    y, fa, T1, T2, am = synthetic(4 * 5 * 3, seed=4)
    shape = (4, 5, 3)
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    paths = {}
    for name, vol in (("mag", np.abs(y)), ("phase", np.angle(y))):
        paths[name] = str(tmp_path / f"{name}.nii.gz")
        nib.save(nib.Nifti1Image(vol.reshape(*shape, len(PHI)).astype(np.float32), aff), paths[name])
    mask = np.ones(shape)
    mask[0, 0, 0] = 0
    paths["mask"] = str(tmp_path / "mask.nii.gz")
    nib.save(nib.Nifti1Image(mask, aff), paths["mask"])
    paths["b1"] = str(tmp_path / "b1.nii.gz")
    nib.save(nib.Nifti1Image((fa / 20.0).reshape(shape), aff), paths["b1"])
    paths["b1_percent"] = str(tmp_path / "b1_percent.nii.gz")
    nib.save(nib.Nifti1Image((100 * fa / 20.0).reshape(shape), aff), paths["b1_percent"])
    return paths, T1.reshape(shape), T2.reshape(shape), mask > 0, tmp_path


def test_fit_nifti(nifti_case):
    p, T1, T2, mask, tmp = nifti_case
    out, res = t2greps.fit_nifti(p["mag"], p["phase"], TR, 20.0, PHI, str(tmp / "out"), mask=p["mask"],
                                 b1=p["b1_percent"], diagnostics=True, verbose=False)  # default: percent
    assert set(out) == {"T1", "T2", "amplitude", "no_bound_mask", "cost", "n_iter", "status"}
    nb = nib.load(out["no_bound_mask"])
    assert nb.get_data_dtype() == np.uint8
    assert np.array_equal(np.asanyarray(nb.dataobj)[mask] == 1, res.no_bound)
    t1 = nib.load(out["T1"]).get_fdata()
    assert t1[0, 0, 0] == 0  # outside mask
    # float32 storage of the inputs limits the recovery accuracy.
    assert np.max(np.abs(t1[mask] - T1[mask]) / T1[mask]) < 1e-3
    assert np.max(np.abs(nib.load(out["T2"]).get_fdata()[mask] - T2[mask]) / T2[mask]) < 1e-4


def test_b1_units_percent_equals_fraction(nifti_case):
    p, T1, T2, mask, tmp = nifti_case
    _, a = t2greps.fit_nifti(p["mag"], p["phase"], TR, 20.0, PHI, str(tmp / "a"), mask=p["mask"], b1=p["b1"],
                             b1_units="fraction", verbose=False)
    _, b = t2greps.fit_nifti(p["mag"], p["phase"], TR, 20.0, PHI, str(tmp / "b"), mask=p["mask"],
                             b1=p["b1_percent"], b1_units="percent", verbose=False)
    assert np.allclose(a.T2, b.T2, rtol=1e-9) and np.allclose(a.T1, b.T1, rtol=1e-9)
