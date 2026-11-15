---
paths:
  - "tests/**/*.py"
---

# Tests in `tests/`

- The layout: `unit/<package>/test_<module>.py`, one file per module; `integration/`, several stages
  together on synthetic MISTRAL files; `e2e/`, the `tolomeo` command on a YAML file; `smoke_test.py`, a script that
  `spin smoke` runs against the installed wheel and sdist. The smoke test must not import `tests.common`: run as a
  script, only its own folder is on the path.
- File names must be unique across the folders: the subfolders have no `__init__.py`, so two files called
  `test_scan.py` collide. That is why the tests of `io` are called `test_io_*.py` and those of `plotting`
  `test_plot_*.py`.
- Shared constants and helpers are in `tests/common.py`: `make_scan` (a synthetic raster with white noise),
  `with_signal`, `gaussian_source`, `pixelized_source`, `write_subscan`, `write_beammap`, `write_scan_dir`,
  `write_skydip`, `write_run` and `RUN_YAML`. Use them before writing new ones. Each file starts with its imports and
  its own constants, then one section per function under test.
- The configuration is in `[tool.pytest]` of `pyproject.toml`: `strict = true` (parametrize ids must be unique, xfail
  is strict) and `filterwarnings = ["error"]`. A test that expects a warning wraps the call in `pytest.warns`; one that
  tolerates a warning it does not test uses `@pytest.mark.filterwarnings`. `--run-slow` (`spin test -s`) also runs the
  tests marked `slow`.
- Random numbers come from `np.random.default_rng(seed)` with a fixed seed, never from the global state.
- Keep each test to a small problem that runs in seconds. `read_scan` and `read_skydip` start a pool of processes, which
  costs about a second per call with `spawn` (macOS): keep those calls few and the synthetic scans short.
- Test with a non-square map (`nx != ny`), so that a swap of the two axes shows up, with NaN in the input, with a fully
  masked selection, and with even and odd sizes.
- Prefer a known answer: a closed form, an injected signal that must come back, a slow numpy reference on a small
  problem, or an independent library (astropy for the WCS). State the tolerance and why.
- To test the accuracy of a map, sample the sky at the centres of the pixels (`pixelized_source`), so that the TOD is
  exactly `P m` plus noise. A sky that changes inside a pixel adds the pixelization error, which the GLS does not
  remove.
- On the short scans of `make_scan` the CG does not get much below a relative residual of `1e-2` when the TOD is only
  noise: the tests of the command line use `tol: 5e-2`.
- Close the figures (`plt.close(fig)`). The plot style uses LaTeX: a test that saves a figure is skipped when `latex`
  is not installed.
- The `tolomeo` logger writes to the current stdout: read its messages with `capsys`, and restore its level after a
  test that runs `main` with `-v` or `-q`.
