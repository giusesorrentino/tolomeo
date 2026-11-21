# tolomeo

`tolomeo` is a Python map-maker for MISTRAL, the camera of the Sardinia Radio Telescope. It is a rewrite of the Fortran
map-maker ROMA (de Gasperis et al. 2016): from the time-ordered data (TOD) of many detectors it builds a sky map by
generalized least squares (GLS), solving the normal equations with a preconditioned conjugate gradient (CG). An earlier
version had a Rust kernel; it was removed, and everything now runs in Python with numpy and scipy.

Personal preferences go in `CLAUDE.local.md` at the root of the repository and in `.claude/settings.local.json`,
both ignored by git. The rules in `.claude/rules/` load when you touch the Python source or the tests.

## Layout

```
src/tolomeo/
  __init__.py      the "tolomeo" logger (INFO on stdout), tolomeo.debug() and tolomeo.info()
  _typing.py       Literal aliases: Projection, Frame, PlotAxes, CommonModeMethod, ResidualNorm
  tod/             TimeOrderedData, Scan, ScanSpec
  coords/          Coordinates; transforms (rot, shift_boresight, radec2xy, xy2latlon, radec2latlon)
  map/             MapGeometry (to_wcs, from_scan, from_beam), MapResult, BinnedMapResult
  io/              read_scan, read_skydip, write_map, read_map
  processing/      process_tod, clean_subscans, FilteringConfig and its steps, masks, skydip calibration
  mapmaking/       public: gls_map, binned_map, estimate_inverse_noise, InverseNoise
                   private: _pointing (pixel of every sample), _noise (inverse spectra), _blocks
                   (overlap-save geometry), _operators (P, N^-1, A, b, preconditioner), _solver (CG),
                   _maps (the two maps)
  plotting/        plot_map, plot_psd, and the matplotlib style in resources/
  _signal/         private numerics: detrend, filters and their gain, common mode, splines
  _lib/            private helpers: units (pydantic aliases), messages (_warn), parallel, spectral
  _cli/            the `tolomeo` command: config.py (RunConfig), runner.py (RunMapmaker), main.py
tests/             unit/<package>/, integration/, e2e/, smoke_test.py; helpers in common.py
.tolomeo/          git-ignored work area: YAML configs, outputs, scratch scripts, notebooks
```

## Commands

- `uv sync` installs the package in editable mode, with the `dev` group (pytest, ruff, pre-commit, spin).
- The local environment is Python 3.15, free-threaded (`.python-version`); the package supports Python 3.12 and later.
- `spin test [paths] [-v] [-s]` runs pytest through uv; `-s` also runs the tests marked `slow`.
- `spin lint [path] [--fix]` runs `ruff check` and then `ruff format` on the path (default `src/tolomeo`), so it
  rewrites files.
- `spin smoke` builds a wheel and an sdist and runs `tests/smoke_test.py` against each, in an isolated environment.
- `tolomeo run.yaml [-v | -q]` makes a map from a YAML file (the keys are documented in `_cli/config.py`). Exit code
  0 if the map was written, 2 if the configuration cannot be read, 1 if the run fails.
- pre-commit runs the file hygiene hooks, zizmor, codespell (`--write-changes`), `ruff check --fix`, `ruff format`,
  mdformat and sp-repo-review. Commit messages follow Conventional Commits: `feat:`, `fix:`, `refactor:`, `test:`,
  `docs:`, `style:`, `chore:`.

## How the pieces fit

`io.read_scan` gives a `Scan` → `processing.calibrate` (optional, from the skydips) → `processing.process_tod`
(baseline, common mode, filters; a mask keeps the source out of the fits) → `MapGeometry` →
`mapmaking.estimate_inverse_noise` (Welch inverse spectrum of every detector, then the filters and the
marginalization) → `mapmaking.gls_map`: `_pointing_matrix`, the flagged and outside samples sent to a virtual pixel,
`_normal_equations`, `_conjugate_gradient` → `MapResult` → `io.write_map`. `binned_map` averages the samples of every
pixel, for checks. The command line runs the same chain in `_cli/runner.py`. Notation: `d = P m + n`,
`A = Pᵀ N⁻¹ P`, `b = Pᵀ N⁻¹ d`.

## Conventions that must not drift

- The TOD has shape `(n_samples, n_det)`. A "sample" is a time instant (a row), not one number of the table.
- `tod.flags` has shape `(n_samples,)` and `True` means a good sample; `valid` has shape `(n_samples, n_det)` and `True`
  also means good. `tod.segments` holds the first row of every subscan and, at the end, `n_samples`.
- `_pointing_matrix` returns `pixels` (`uintp`, `0` for the samples outside the map) and `valid`: use `pixels` only
  together with `valid`.
- Noise model (`InverseNoise`): `inv_noiseps` has shape `(n_freqs, n_cols)` with `n_freqs = nperseg // 2 + 1`;
  `det_pairs` has shape `(n_cols, 2)` and dtype `uintp`; `lag_zero` has shape `(n_det,)`. Only auto spectra:
  `n_cols = n_det`, and `cross_spectra=True` raises `NotImplementedError`. `nperseg` is even, and the same length is
  used by the Welch blocks of the estimate and by the filter blocks of `gls_map`.
- Overlap-save blocks (`_blocks.py`): `block_len = nperseg`, `core_len = nperseg // 2`, `context_len = core_len // 2`.
- Maps have shape `(ny, nx)`: the first index is y, the flat pixel is `iy * nx + ix`. `MapResult` and
  `BinnedMapResult` check the shapes in `model_post_init`.
- Angles are radians inside the package; `MapGeometry.pixel_size` is in arcsec. FITS headers use degrees, and `CRPIX`
  is 1-based. A configuration value with a physical dimension has an implicit unit and also accepts a string with a
  unit (`"4 arcsec"`, `"50 mHz"`), through the aliases of `_lib/units.py`.
- Messages: progress and timings are `logger.info` with a tag (`[POINTING] ...`, `[NOISE] ...`, `[MAP] ...`); a problem
  the user can fix is a warning issued with `_lib/messages._warn`; an error is an exception. Details in
  `.claude/rules/python.md`.
- Public and private names follow SciPy's convention; how it is applied here is in `.claude/rules/python.md`.

## Scientific facts already established

- A raw scan cannot go straight into the GLS: the atmosphere (a common mode) and the offsets of the detectors are
  amplified 30 to 80 times in the map. Clean the TOD first.
- Cross spectra stored as the element-wise `1 / Re(csd)` do not make a positive definite operator; how to use them is
  an open question, and they are not implemented.
- The large scales converge slowest in the CG: raise `maxiter` before suspecting a bug.
- A sky that changes inside a pixel adds a pixelization error that the GLS does not remove, larger with non-flat
  weights: keep the pixel well below the beam.
- Open question, not an established cause: the zero weight at `f = 0` (`_inverse_spectra`, `_shape_inverse`) can
  stall the CG on faint fields. The numbers are in the skill `diagnose-mapmaking`.

## Tools in this folder

- `diagnose-mapmaking` loads by itself when a map looks wrong.
- `/guide`, `/explain` and `/critique` are the tutoring skills, and `Tutor` is the matching output style; they are
  described in `.claude/README.md`.
