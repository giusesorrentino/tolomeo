---
paths:
  - "src/tolomeo/**/*.py"
---

# Python code in `src/tolomeo/`

- Public and private names follow SciPy's convention ("Importing from SciPy"). In Python the line between the public API
  and the private implementation details is not always clear, and nothing enforces it: a private name can be accessed,
  but code that does so may break without warning in a future release. The rules:
  - A function, class, method or module attribute whose name starts with an underscore is private.
  - If a class name starts with an underscore, none of its members is public, whether or not their own names start with
    an underscore.
  - If a module name in a package starts with an underscore, none of its members is public, whether or not their own
    names start with an underscore.
  - If a module or package defines `__all__`, that authoritatively defines the public interface.
  - If a module or package does not define `__all__`, every name that does not start with an underscore is public.
  - An underscore marks something as private, but the absence of an underscore does not mark it as public: a module
    whose name has no underscore can still be private, so the public API is declared explicitly and not guessed from the
    names.
- How these rules are applied in this package:
  - Every module of a public subpackage starts with an underscore (`io/_scan.py`,
    `map/_results.py`, `processing/_filtering.py`), as in SciPy: the public names are
    only the ones that the package `__init__.py` re-exports. Code and tests import a
    public name from the package (`from tolomeo.processing import RadiusMask`), and a
    private one from its module (`from tolomeo.io._map import check_fits_suffix`).
  - A function or class that is part of the public API keeps a plain name and is listed in the `__all__` of the module
    that defines it, **also when that module is private**, and in the `__all__` of the package `__init__.py` that
    re-exports it. In `mapmaking`: `_maps.py` has `__all__ = ["binned_map", "gls_map"]`, and
    `mapmaking/__init__.py` has `__all__ = ["InverseNoise", "binned_map", "estimate_inverse_noise", "gls_map"]`.
  - That `__all__` lists **only** the public names. Every other function, type alias and constant of the module
    starts with an underscore and is never in an `__all__`, even when other modules import it (`_pointing_matrix`,
    `_bin`, `_conjugate_gradient`).
  - Classes are the exception: a class name never starts with an underscore, also when the class is private
    (`BlockLayout`, `NormalOperator`, `CGOperator`). Whether a class is public is decided only by `__all__`: a private
    class lives in a private module and is in no `__all__`. Do not add an underscore to a class name.
  - A module that defines no public name has no `__all__` (`_blocks.py`, `_operators.py`, `_pointing.py`, `_solver.py`).
  - Not every module follows these rules yet (for example `InverseNoise` is missing from the `__all__` of `_noise.py`).
    The user knows and will fix them: do not change an `__all__` or rename anything the task does not need, and do
    not point these mismatches out unless the task changes the public names of that module.
- Docstrings use the NumPy style (`Parameters`, `Returns`, `Yields`, `Attributes`, `Notes`, ...). Every public function
  and class has one, and so do the private ones that are not obvious. The rules:
  - A docstring of more than one line starts on the line after the opening `"""`; only a one-line docstring stays on
    one line.
  - The `name : type` line holds only the type, plus `optional` when the argument can be omitted. The shape, the
    default and the units go in the description of the parameter, never on the type line.
  - The shape and the default are the last sentences of the description and go at the end of its last line when
    they fit there within 88 columns. Only when they do not fit do they start a new line, whole: "Shape" or
    "Defaults to" is never separated from its value, and a literal in double backticks is never split across two
    lines. The default is written "Defaults to ``value``.", never "Default ``value``." or "Default to".
  - Every array gives its dtype (on the type line) and its shape (in the description), and units when they matter.
  - A class is written as a Sphinx role with its full import path and a tilde: `` :class:`~numpy.ndarray` ``,
    `` :class:`~pathlib.Path` ``, `` :class:`~astropy.wcs.WCS` ``, `` :class:`~tolomeo.map.MapResult` ``. The aliases
    of `tolomeo/_typing.py` are written the same way (`` :class:`~tolomeo._typing.Projection` ``) instead of listing
    their values. Builtins (`int`, `float`, `bool`, `str`, `dict`, `list`, `tuple`) stay plain.
  - Example:

    ```python
    def f(tod: NDArray[np.float64], order: int = 1, background: NDArray[np.bool_] | None = None) -> NDArray[np.float64]:
        """
        Short summary on the line after the quotes.

        Parameters
        ----------
        tod : :class:`~numpy.ndarray` of float64
            Cleaned TOD, in the units of the instrument. Shape ``(n_samples, n_det)``.
        order : int
            Order of the filter. Defaults to ``1``.
        background : :class:`~numpy.ndarray` of bool, optional
            True for the samples far from the source. Shape ``(n_samples,)``. Defaults to
            ``None``, all the samples.

        Returns
        -------
        :class:`~numpy.ndarray` of float64
            The filtered TOD. Shape ``(n_samples, n_det)``.
        """
    ```
- Type hints use `NDArray[np.float64]` and similar; abstract containers come from `collections.abc` (`Sequence`, not
  `typing.Sequence` and not `list` when any sequence is accepted). The package supports Python 3.12 and later: an
  annotation that names a class defined further down in the same module needs `from __future__ import annotations`,
  otherwise Python 3.12 and 3.13 raise `NameError` at import time. The local environment is Python 3.15, which
  evaluates annotations lazily: there the `NameError` never shows up, so passing local tests prove nothing about it.
- Ruff: line length 88, with the rules `B`, `I` and `UP` on top of the defaults (`[tool.ruff.lint]` in
  `pyproject.toml`). `zip` gets an explicit `strict=`: `True` when the two sequences must have the same length.
- A class that only holds data (fields, plus small properties or helper methods) is a pydantic `BaseModel`, since
  pydantic is a dependency of the package; use `dataclasses` only where pydantic is not available. A class with real
  behaviour (an operator, a solver) is a plain class, not a data class. For pydantic models: numpy fields need
  `model_config = ConfigDict(arbitrary_types_allowed=True)` (arrays are checked with `isinstance`, not copied, and
  their dtype is not checked); an immutable one adds `frozen=True`; a copy with changes is `model_copy(update=...)`
  (not `dataclasses.replace`); checks after construction go in `model_post_init(self, context, /)`; the constructor
  takes keyword arguments only.
- Comments and docstrings are in English. A few older docstrings are in Italian: do not translate them unless the user
  asks.
- Validate at the top of a public function and raise `ValueError` with the shapes in the message
  (`f"expected (n_samples, n_det), got {tod.shape}"`). Decide the NaN policy explicitly instead of letting NaN travel.
- Never modify an input array in place; copy first, and say so in the docstring when a copy is made. When a function
  offers `overwrite_scan=True`, the docstring says what is modified.
- Messages follow the table "When to use logging" of the Python logging HOWTO:
  - a problem the user can fix (a parameter, a mask, the configuration) is a warning, issued with `_warn(message,
    category)` from `_lib/messages.py`, which points it to the first line of code outside tolomeo. In a loop the text
    stays the same at every pass, so that Python shows it once, and the numbers go to a `logger.debug` message;
  - progress and timings are `logger.info` on `logger = logging.getLogger(__name__)`, with the tag in the text
    (`[NOISE] ...`); details are `logger.debug`. The `tolomeo` logger shows INFO on stdout by default
    (`tolomeo/__init__.py`); `tolomeo.debug()` and `tolomeo -v` show DEBUG too, `tolomeo -q` only the warnings;
  - `print` only for the messages of the command line itself (`error: ...`, `warning: ...` in `_cli/main.py`);
  - an error is an exception.
- `read_scan` and `read_skydip` read the FITS files in a `ProcessPoolExecutor`: the function that runs in a worker must
  be defined at the top level of the module, because macOS starts the workers with `spawn`.
- When a public name, signature or default changes, update `__all__` and the package `__init__.py`, and tell the user.
  Do not change the default of a scientific parameter (`tol`, `maxiter`, `nperseg`, `min_hits_frac`, `cross_spectra`)
  silently.
- Vocabulary: `n_samples`, `n_det`, `nperseg`, `n_freqs`, `n_cols`, `det_pairs`, `inv_noiseps`, `lag_zero`, `pixels`,
  `valid`, `geometry`, `core_len`, `context_len`. If a new name is needed, propose it and wait for the user's decision.
- Reuse what exists (`Coordinates`, `MapGeometry`, `_pointing_matrix`, `_bin`) before suggesting new helpers, and check
  with a search that a function is really unused before saying so.
