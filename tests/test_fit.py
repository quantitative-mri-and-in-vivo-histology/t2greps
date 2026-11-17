"""Fit: recovery, derivatives, bounds, threads, NIfTI I/O and the MagPhsT2-compatible wrapper."""

import nibabel as nib
import numpy as np
import pytest

import t2greps
from t2greps.compat import cal_T2T1AM
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
                                 b1=p["b1"], diagnostics=True, verbose=False)
    assert set(out) == {"T1", "T2", "amplitude", "cost", "n_iter", "status"}
    t1 = nib.load(out["T1"]).get_fdata()
    assert t1[0, 0, 0] == 0  # outside mask
    # float32 storage of the inputs limits the recovery accuracy.
    assert np.max(np.abs(t1[mask] - T1[mask]) / T1[mask]) < 1e-3
    assert np.max(np.abs(nib.load(out["T2"]).get_fdata()[mask] - T2[mask]) / T2[mask]) < 1e-4


def test_compat_cal_T2T1AM(nifti_case):
    p, T1, T2, mask, tmp = nifti_case
    cal_T2T1AM(p["mag"], p["phase"], p["mask"], p["b1_percent"], TR, 20.0, PHI, outputdir=str(tmp))
    for name in ("T1_", "T2_", "Am_"):
        assert (tmp / f"{name}.nii.gz").exists()
    t2 = nib.load(tmp / "T2_.nii.gz").get_fdata()
    assert np.max(np.abs(t2[mask] - T2[mask]) / T2[mask]) < 1e-4
