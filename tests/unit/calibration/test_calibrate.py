import numpy as np
import pytest

from tests.common import make_scan
from tolomeo.calibration import calibrate

TAU_ATM = 0.16
RESPONSIVITY = np.array([-2.0, -1.0, 0.5, 3.0])  # signal units per kelvin


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
    assert scan.tod.unit == "rad"


@pytest.mark.parametrize("unit", ["K", "mK"])
def test_calibrate_twice_raises(unit):
    # A TOD already in a temperature would be divided by the responsivity again.
    scan = make_scan()
    scan = scan.model_copy(update={"tod": scan.tod.model_copy(update={"unit": unit})})

    with pytest.raises(ValueError, match=r"already in"):
        calibrate(scan, RESPONSIVITY)


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
