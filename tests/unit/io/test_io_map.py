from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from tests.common import DEC0, RA0
from tolomeo.io import read_map, write_map
from tolomeo.io.map import check_fits_suffix
from tolomeo.map import MapGeometry, MapResult

rng = np.random.default_rng(3)

NX, NY = 6, 4


def _map_result(frame="radec", projection="sin", weights=True, unit="K"):
    """
    A MapResult with random maps on a small grid. The solver is ``"minres"``, not the
    default of ``gls_map``, so that a reader that ignores ``SOLVER`` shows up.
    """
    geometry = MapGeometry(
        nx=NX,
        ny=NY,
        pixel_size=12.5,
        center=(RA0, DEC0),
        projection=projection,
        frame=frame,
    )
    return MapResult(
        image=rng.standard_normal((NY, NX)),
        hits=rng.integers(0, 100, (NY, NX)),
        n_iter=17,
        residual=3.5e-7,
        geometry=geometry,
        solver="minres",
        weights=rng.random((NY, NX)) if weights else None,
        converged=False,
        unit=unit,
    )


def _assert_same_map(read, result):
    np.testing.assert_array_equal(read.image, result.image)
    np.testing.assert_array_equal(read.hits, result.hits)
    if result.weights is None:
        assert read.weights is None
    else:
        np.testing.assert_array_equal(read.weights, result.weights)
    assert read.n_iter == result.n_iter
    assert read.residual == result.residual
    assert read.converged == result.converged
    assert read.solver == result.solver
    assert read.unit == result.unit

    geometry, expected = read.geometry, result.geometry
    assert (geometry.nx, geometry.ny) == (expected.nx, expected.ny)
    assert np.isclose(geometry.pixel_size, expected.pixel_size, rtol=1e-12)
    np.testing.assert_allclose(geometry.center, expected.center, rtol=1e-12)
    assert geometry.projection == expected.projection
    assert geometry.frame == expected.frame


# -----------------------------------------------------------------------------
# check_fits_suffix
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["map.fits", "map.FIT", "map.fts", "map.fits.gz"])
def test_check_fits_suffix_valid(name):
    check_fits_suffix(Path(name))


@pytest.mark.parametrize("name", ["map.txt", "map", "map.fits.zip"])
def test_check_fits_suffix_invalid_raises(name):
    with pytest.raises(ValueError, match=r"must end with one of"):
        check_fits_suffix(Path(name))


# -----------------------------------------------------------------------------
# write_map / read_map
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("projection", ["sin", "tan", "zea"])
def test_map_round_trip_radec(tmp_path, projection):
    result = _map_result(projection=projection)
    path = tmp_path / "map.fits"

    write_map(result, path)

    _assert_same_map(read_map(path), result)


def test_map_round_trip_azel(tmp_path):
    # Check that the centre and the projection of an azel map, which are not in its
    # WCS, are read back from FRAME, OBSRA, OBSDEC and PROJ.
    result = _map_result(frame="azel", projection="tan")
    path = tmp_path / "map.fits"

    write_map(result, path)

    _assert_same_map(read_map(path), result)


def test_map_round_trip_without_weights_and_unit(tmp_path):
    result = _map_result(weights=False, unit=None)
    path = tmp_path / "map.fits.gz"

    write_map(result, path)

    _assert_same_map(read_map(path), result)


def test_write_map_header_and_extensions(tmp_path):
    path = tmp_path / "map.fits"

    write_map(
        _map_result(),
        path,
        extra_header={"OBJECT": "3C84"},
        config="name: run\nnote: però\n",
    )

    with fits.open(path) as hdul:
        assert [hdu.name for hdu in hdul] == ["SCI", "HITS", "WEIGHTS", "CONFIG"]
        assert hdul["SCI"].header["NITER"] == 17
        assert hdul["SCI"].header["SOLVER"] == "minres"
        assert hdul["SCI"].header["BUNIT"] == "K"
        assert hdul["HITS"].header["BUNIT"] == "count"
        assert all(hdu.header["OBJECT"] == "3C84" for hdu in hdul[:3])
        # the characters that are not ASCII are written as escape sequences
        assert list(hdul["CONFIG"].data["LINE"]) == ["name: run", "note: per\\xf2"]


def test_read_map_without_solver_is_cg(tmp_path):
    # A file written before SOLVER existed was made with the CG, the only solver then.
    path = tmp_path / "map.fits"
    write_map(_map_result(), path)
    fits.delval(path, "SOLVER", ext=0)

    assert read_map(path).solver == "cg"


def test_write_map_does_not_overwrite(tmp_path):
    path = tmp_path / "map.fits"
    write_map(_map_result(), path)

    with pytest.raises(OSError):
        write_map(_map_result(), path)

    write_map(_map_result(), path, overwrite=True)


def test_write_map_invalid_suffix_raises(tmp_path):
    with pytest.raises(ValueError, match=r"must end with one of"):
        write_map(_map_result(), tmp_path / "map.npy")


def test_read_map_not_a_tolomeo_map_raises(tmp_path):
    # Check that a FITS image without the keywords of tolomeo is refused.
    path = tmp_path / "other.fits"
    fits.PrimaryHDU(np.zeros((NY, NX))).writeto(path)

    with pytest.raises(ValueError, match=r"is not a tolomeo map: missing"):
        read_map(path)


def test_read_map_without_data_raises(tmp_path):
    path = tmp_path / "empty.fits"
    fits.PrimaryHDU().writeto(path)

    with pytest.raises(ValueError, match=r"the primary HDU has no data"):
        read_map(path)
