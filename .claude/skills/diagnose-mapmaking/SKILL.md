---
name: diagnose-mapmaking
description: Use when a map-making or imaging result looks wrong (a blob, stripes, huge amplitude, a missing source, almost all samples outside the map, a CG that does not converge) to find the cause step by step.
---

# Diagnose a bad map

Go in this order. Each step is cheap and rules out a whole class of causes. While you go, say what each check is for and how
the user can repeat it. When the cause is found, name the fix; if the output style `Tutor` is active, name it in words
and let the user write it.

0. **Find out what was run.** The header of the `SCI` extension of the FITS map has `NITER`, `RESID` and `CONVERGD`; a map
   made with the command line also has `NPERSEG`, `GLSTOL` and `MAXITER`, and the YAML of the run, line by line, in the
   `CONFIG` extension. Check: raw or cleaned TOD, the filtering steps and the mask, the filters of the inverse spectra,
   the geometry (centre, pixel size, projection, frame), the solver tolerance and the iteration limit. A diagnosis for
   the wrong configuration wastes time.
1. **Pointing.**
   - Units: radians versus degrees, arcsec versus degrees. Inside the package angles are radians and `pixel_size` is arcsec.
   - Fraction of samples outside the map: `gls_map` logs it (`[POINTING] ... fall outside the map`). A large fraction is
     normal if the scan is bigger than the map; 100% means a wrong centre, and `gls_map` stops with `No valid sample`.
   - NaN in the coordinates: `Coordinates.center` ignores them, but `process_tod` drops the samples with a NaN pointing.
1. **Binned map** of the same cleaned TOD (`binned_map`). Is the source where it should be? Check its sign (detectors that
   read a frequency shift can make a source negative). Use a robust colour scale (percentiles 1 to 99.5): extreme values
   elsewhere can hide the source.
1. **TOD cleaning.** Raw scans carry the atmosphere (a common mode with a large amplitude) and the offsets of the detectors.
   With auto spectra only, a GLS map amplifies the large-scale modes that the noise model barely constrains. Warning sign:
   the map range is 10 to 100 times the TOD range. Clean first (`process_tod`: baseline and common mode, with a mask on the
   source), then map. Compare the binned maps before and after the cleaning to see what the cleaning removed.
1. **Noise model.** Plot the inverse spectra (`InverseNoise.freqs`, `inv_noiseps`) and the PSD of the cleaned TOD
   (`plot_psd`). Check that `nperseg` fits in the segments of good samples (`nperseg: auto` chooses it from them), that the
   passband of the marginalization is the intended one (`InverseNoise.passband`), and that no detector has a spectrum far
   from the others. Only auto spectra are implemented: cross spectra stored as the element-wise `1 / Re(csd)` would not
   make a positive definite operator.
1. **Solver.** Read the residual log, and `CONVERGD` in the header (`gls_map` also warns when the CG stops at `maxiter`).
   The large scales converge slowest: raise `maxiter` before suspecting a bug. Known open issue: on faint fields the CG
   often stops above `tol` (A1995: 11 maps of 23 did not converge in 400 iterations); on synthetic data it stalls when
   the weight at `f = 0` is zero and the spectra are noisy, and converges when that weight is not zero. Say that this
   is an open question, not an established cause.
1. **Pixelization.** On a bright source, compare the peak of the GLS map with the peak of the binned map. If the GLS
   peak is clearly lower, the sky changes inside a pixel and the GLS spreads that error (on a synthetic raster, a 30
   arcsec source lost 31% of its peak with 10 arcsec pixels and 3.5% with 2.5 arcsec pixels). Try smaller pixels.
1. **Implementation bug or problem set-up?** Build a dense reference on a small problem: a few detectors, a coarse map,
   `A` as a dense matrix (apply the `project` method of the operator returned by `_normal_equations` to the columns of
   the identity; there are `n_unknowns` of them, the sky pixels plus the virtual pixel) and `b`; solve it with
   `numpy.linalg` and compare with the CG. A sky constant inside every pixel (`tests/common.py`, `pixelized_source`)
   must come back exactly.
   If the reference shows the same problem, the cause is the data or the set-up, not the code.
1. **Orientation and WCS.** Project known (RA, Dec) points with `_pointing_matrix` and with `MapGeometry.to_wcs`: they
   must land in the same pixel. Check the sign of the RA axis (east on the left) and, for `frame="azel"`, the rotation by
   the parallactic angle.

## Always report

The numbers behind each conclusion (ranges, fractions, residuals, iterations), and what you could not reproduce. Save plots
in `.tolomeo/` and say the file name.
