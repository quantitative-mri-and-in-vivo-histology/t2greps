"""t2greps: T2 (and T1, amplitude) mapping from gradient echo with partial RF spoiling (GREPS).

    import t2greps
    s = t2greps.signal(T1=1000, T2=80, flip_angle=20, phase_increment=2, tr=10.8)
    res = t2greps.fit(data, flip_angle, tr=10.8, phase_increments=[1, 2, 4, 1.5, 3, 5])
    paths, res = t2greps.fit_nifti(mag, phase, tr=10.8, flip_angle=20,
                                   phase_increments=[1, 2, 4, 1.5, 3, 5], output_dir="out",
                                   mask=mask, b1=b1, b1_units="fraction")
"""

from .epg import signal, signal_stepped
from .fit import FitResult, Status, fit
from .io import fit_nifti, load_data

__version__ = "0.1.0"
__all__ = ["signal", "signal_stepped", "fit", "FitResult", "Status", "fit_nifti", "load_data"]
