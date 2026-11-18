# t2greps

T2 (and T1, amplitude) mapping from **g**radient **e**cho with **p**artial RF **s**poiling (GREPS):
magnitude and phase of GRE images acquired with several quadratic RF phase increments.

A replacement for [MagPhsT2](https://github.com/mrphysics-bonn/MagPhsT2) (`T2T1AM.py`). It uses the same
signal model, data, start values and bounds, and computes the steady state exactly instead of
simulating TR by TR. This reduces the fitting time from hours to minutes, or even seconds.

## Install

```bash
pip install path/to/t2greps        # or: pip install -e . for development
```

## Command line

```bash
t2greps-fit magnitude.nii.gz phase.nii.gz \
    --tr 10.8 --flip-angle 20 --phase-increments 1 2 4 1.5 3 5 \
    --mask brain_mask.nii.gz --b1 B1_map.nii.gz --b1-units fraction \
    -o out/
```

Inputs:

* `magnitude`, `phase`: 4D, one volume per phase increment, **background-corrected** (as produced by the
  traveling-heads preprocessing: the ±increment pairs combined). Phase in radians.
* `phase_increments`: degrees, **in volume order**.
* `b1`: actual / nominal flip angle **in percent** (100 = nominal, default), or as a fraction
  (1 = nominal) with `--b1-units fraction`.
* `tr`: time between excitations in ms (`RepetitionTimeExcitation` × 1000).

Outputs, written to the output directory (with `--prefix` in front of each name):

* `T1.nii.gz`, `T2.nii.gz`: relaxation times in ms.
* `amplitude.nii.gz`: signal amplitude A (proton density and receive sensitivity).
* `no_bound_mask.nii.gz`: 1 where no parameter ended on one of its bounds (T1 1–10000 ms, T2 1–1000 ms,
  amplitude 1–50000 by default). Where a bound is hit the values are less reliable. With T2 at its lower
  bound the signal no longer depends on T1, so T1 and amplitude are undetermined there. Use the mask to
  exclude such voxels from statistics.
* `cost.nii.gz`, `n_iter.nii.gz`, `status.nii.gz`: only with `--diagnostics`.

## Python API

`fit` works on arrays, and `signal` evaluates the signal model.

```python
import t2greps

# arrays: data = magnitude * exp(1j * phase), shape (n_voxels, n_increments)
res = t2greps.fit(data, flip_angle_per_voxel, tr=10.8, phase_increments=[1, 2, 4, 1.5, 3, 5])
res.T1, res.T2, res.amplitude, res.no_bound, res.cost, res.n_iter, res.converged

# signal model
s = t2greps.signal(T1=1000, T2=80, flip_angle=20, phase_increment=2, tr=10.8)   # complex
```

## Method

As in Wang et al. (2024), T1, T2 and an amplitude A are fitted by minimizing
Σ|S_meas − A·S(T2, T1)|² over the phase increments. The model signal S is the steady state of the
extended phase graph (EPG) (Weigel, 2015) with RF phase φₙ = Δφ·n(n+1)/2.

[MagPhsT2](https://github.com/mrphysics-bonn/MagPhsT2) simulates the EPG with
[EPGpp](https://github.com/mrphysics-bonn/EPGpp) TR by TR until the signal stops changing. t2greps computes the steady state
directly. Following Ganter (2006), the magnetization at each dephasing order k is rotated by a phase
that depends on n and k. Then every TR applies the same map x ↦ Mx + b, and the steady state is
x = (I − M)⁻¹b. It is computed by a recursion from K = max(16, ⌈12·T2/TR⌉) down to k = 0, with a
relative error below 1e-11.

A follows in closed form for given T1 and T2. T1 and T2 are fitted with Levenberg–Marquardt (Marquardt, 1963) and exact
derivatives, with voxels in parallel.

## References

* Ganter C. Steady state of gradient echo sequences with radiofrequency phase cycling: analytical
  solution, contrast enhancement with partial spoiling. *Magn Reson Med* 2006;55:98–107.
  [doi:10.1002/mrm.20736](https://doi.org/10.1002/mrm.20736)
* Marquardt DW. An algorithm for least-squares estimation of nonlinear parameters.
  *J Soc Ind Appl Math* 1963;11:431–441. [doi:10.1137/0111030](https://doi.org/10.1137/0111030)
* Wang D, Stirnberg R, Stöcker T. Improved gradient echo magnitude- and phase-based mapping of T2
  using multiple RF spoiling increments at 3T and 7T. *Magn Reson Med* 2024;92:2328–2342.
  [doi:10.1002/mrm.30217](https://doi.org/10.1002/mrm.30217)
* Weigel M. Extended phase graphs: dephasing, RF pulses, and echoes – pure and simple.
  *J Magn Reson Imaging* 2015;41:266–295. [doi:10.1002/jmri.24619](https://doi.org/10.1002/jmri.24619)

## License

MIT, see [LICENSE](LICENSE).
