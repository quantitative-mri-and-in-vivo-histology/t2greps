"""Command line: fit GREPS NIfTI images.

    t2greps-fit magnitude.nii.gz phase.nii.gz --tr 10.8 --flip-angle 20 \\
        --phase-increments 1 2 4 1.5 3 5 --mask mask.nii.gz --b1 b1.nii.gz -o out
"""

import argparse

from .io import fit_nifti


def main(argv=None):
    p = argparse.ArgumentParser(prog="t2greps-fit", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("magnitude", help="4D magnitude, one volume per phase increment (background-corrected)")
    p.add_argument("phase", help="4D phase in radians, same layout")
    p.add_argument("--tr", type=float, required=True, help="repetition time between excitations [ms]")
    p.add_argument("--flip-angle", type=float, required=True, help="nominal flip angle [deg]")
    p.add_argument("--phase-increments", type=float, nargs="+", required=True,
                   help="RF phase increments [deg], in volume order")
    p.add_argument("--mask", default=None, help="3D mask; voxels > 0.5 are fitted (default: all)")
    p.add_argument("--b1", default=None, help="3D B1 map (actual / nominal flip angle)")
    p.add_argument("--b1-units", choices=["fraction", "percent"], default="fraction")
    p.add_argument("-o", "--output-dir", required=True)
    p.add_argument("--prefix", default="", help="prefix for output file names")
    p.add_argument("--diagnostics", action="store_true", help="also write cost, n_iter and status maps")
    p.add_argument("--threads", type=int, default=None, help="number of threads (default: all cores)")
    p.add_argument("--max-iter", type=int, default=200)
    a = p.parse_args(argv)
    fit_nifti(a.magnitude, a.phase, a.tr, a.flip_angle, a.phase_increments, a.output_dir, mask=a.mask,
              b1=a.b1, b1_units=a.b1_units, prefix=a.prefix, diagnostics=a.diagnostics,
              n_threads=a.threads, max_iter=a.max_iter)


if __name__ == "__main__":
    main()
