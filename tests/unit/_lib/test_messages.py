import pytest

from tolomeo._lib.messages import _warn

# -----------------------------------------------------------------------------
# _warn
# -----------------------------------------------------------------------------


def test_warn_default_category():
    with pytest.warns(UserWarning, match=r"something to fix"):
        _warn("something to fix")


def test_warn_category():
    with pytest.warns(RuntimeWarning, match=r"did not converge"):
        _warn("did not converge", RuntimeWarning)


def test_warn_points_to_the_code_outside_tolomeo():
    # Check that the warning is attributed to this file, the first frame outside the
    # package, and not to the line of tolomeo that issued it.
    with pytest.warns(UserWarning) as record:
        _warn("where am I")

    assert record[0].filename == __file__
