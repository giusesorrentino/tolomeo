"""
Warnings for the user of tolomeo.

The messages of the package follow the table "When to use logging" of the Python
logging HOWTO:

* a problem that the user can fix (a parameter, a mask, the configuration) is a
  warning, issued with :func:`_warn`;
* progress and timings go to the ``tolomeo`` logger (see ``tolomeo/__init__.py``);
* an error is an exception.
"""

import os
import warnings

# folder of the package, with the final separator: the frames of the code inside it are
# skipped, so that a warning points to the line of the user's code that called tolomeo
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def _warn(message: str, category: type[Warning] = UserWarning) -> None:
    """
    Issue a warning that points to the first line of code outside tolomeo.

    With ``warnings.warn(..., stacklevel=2)`` the warning points to the caller of the
    function, which is often another function of tolomeo. ``skip_file_prefixes``
    (Python 3.12) skips all the frames inside the package instead, at any depth. Python
    shows a warning once for every text and line of code, so a warning issued in a
    loop with the same text is shown once.

    Parameters
    ----------
    message : str
        Text of the warning.
    category : type
        Class of the warning. Default :class:`UserWarning`; :class:`RuntimeWarning` for
        a problem met during a computation (for example a CG that does not converge).
    """
    warnings.warn(message, category, skip_file_prefixes=(_PACKAGE_DIR,))
