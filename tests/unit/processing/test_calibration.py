import numpy as np
import pytest

from tests.common import make_scan
from tolomeo.io.skydip import SkyDip
from tolomeo.processing import calibrate, skydip_responsivity

T_ATM = 267.0
TAU_ATM = 0.16
RESPONSIVITY = np.array([-2.0, -1.0, 0.5, 3.0])  # signal units per kelvin


def _skydip(responsivity=RESPONSIVITY, el=None, offset=5.0):
    """A skydip whose signal is exactly ``offset + responsivity * T_sky``."""
    el = np.linspace(np.deg2rad(20), np.deg2rad(80), 200) if el is None else el
    t_sky = T_ATM * (1 - np.exp(-TAU_ATM / np.sin(el)))
    return SkyDip(el=el, signal=offset + t_sky[:, None] * responsivity[None, :])


# -----------------------------------------------------------------------------
# skydip_responsivity
# -----------------------------------------------------------------------------


def test_skydip_responsivity_recovers_the_slope():
    responsivity = skydip_responsivity([_skydip()], T_atm=T_ATM, tau_atm=TAU_ATM)

    np.testing.assert_allclose(responsivity, RESPONSIVITY)


def test_skydip_responsivity_median_of_the_skydips():
    # Check that one anomalous skydip does not move the result: median, not mean.
    skydips = [_skydip(), _skydip(), _skydip(0.39 * RESPONSIVITY)]

    responsivity = skydip_responsivity(skydips, T_atm=T_ATM, tau_atm=TAU_ATM)

    np.testing.assert_allclose(responsivity, RESPONSIVITY)


def test_skydip_responsivity_constant_elevation_warns():
    # Check that a skydip whose elevation does not change is skipped, with a warning.
    flat = _skydip(el=np.full(200, 0.8))

    with pytest.warns(RuntimeWarning, match=r"1 of 2 skydips skipped"):
        responsivity = skydip_responsivity(
            [flat, _skydip()], T_atm=T_ATM, tau_atm=TAU_ATM
        )

    np.testing.assert_allclose(responsivity, RESPONSIVITY)


def test_skydip_responsivity_no_usable_skydip_raises():
    flat = _skydip(el=np.full(200, 0.8))

    with (
        pytest.warns(RuntimeWarning),
        pytest.raises(ValueError, match=r"No usable skydip"),
    ):
        skydip_responsivity([flat])


# -----------------------------------------------------------------------------
# calibrate
# -----------------------------------------------------------------------------


def test_calibrate_divides_by_the_responsivity():
    scan = make_scan()

    calibrated = calibrate(scan, RESPONSIVITY)

    np.testing.assert_allclose(calibrated.tod.signal, scan.tod.signal / RESPONSIVITY)
    assert calibrated.tod.unit == "K"


def test_calibrate_does_not_modify_the_input():
    scan = make_scan()
    original = scan.tod.signal.copy()

    calibrate(scan, RESPONSIVITY)

    np.testing.assert_array_equal(scan.tod.signal, original)
    assert scan.tod.unit is None


def test_calibrate_opacity_correction():
    # Check that with tau_atm the signal is also divided by the transmission of the
    # atmosphere at the elevation of every sample.
    scan = make_scan()

    calibrated = calibrate(scan, RESPONSIVITY, tau_atm=TAU_ATM)

    transmission = np.exp(-TAU_ATM / np.sin(scan.coordinates.el))[:, None]
    np.testing.assert_allclose(
        calibrated.tod.signal, scan.tod.signal / RESPONSIVITY / transmission
    )


def test_calibrate_wrong_shape_raises():
    with pytest.raises(ValueError, match=r"`responsivity` has shape"):
        calibrate(make_scan(), RESPONSIVITY[:3])


@pytest.mark.parametrize("bad", [0.0, np.nan, np.inf])
def test_calibrate_invalid_responsivity_raises(bad):
    responsivity = RESPONSIVITY.copy()
    responsivity[1] = bad

    with pytest.raises(ValueError, match=r"zero or non-finite responsivity"):
        calibrate(make_scan(), responsivity)
