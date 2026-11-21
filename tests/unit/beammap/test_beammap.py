import numpy as np
import pytest

from tolomeo.beammap import BeamMap, ChannelFlag


def _beammap(**changes):
    """A BeamMap of three channels, with ``changes`` to its fields."""
    fields = {
        "channel_ids": np.array([3, 7, 12]),
        "flags": np.array([0, 2, 0]),
        "lat": np.zeros(3),
        "lon": np.zeros(3),
    }
    return BeamMap(**(fields | changes))


# -----------------------------------------------------------------------------
# BeamMap
# -----------------------------------------------------------------------------


def test_beammap_what_is_not_known_is_none():
    beammap = _beammap()

    for name in [
        "lat_error",
        "lon_error",
        "ref_channel",
        "ref_lat",
        "ref_lon",
        "snr_threshold",
        "source_name",
        "time_range",
    ]:
        assert getattr(beammap, name) is None


@pytest.mark.parametrize("name", ["flags", "lat", "lon", "lat_error", "lon_error"])
def test_beammap_array_of_another_shape_raises(name):
    with pytest.raises(ValueError, match=rf"`{name}` has shape \(2,\)"):
        _beammap(**{name: np.zeros(2)})


def test_beammap_duplicate_ids_raises():
    with pytest.raises(ValueError, match=r"duplicate detector ids"):
        _beammap(channel_ids=np.array([3, 3, 12]))


# -----------------------------------------------------------------------------
# ChannelFlag
# -----------------------------------------------------------------------------


def test_channel_flag_values():
    # Check that every reason is its own bit, so that their sum keeps all of them.
    values = [int(flag) for flag in ChannelFlag]

    assert values == [1, 2, 4, 8, 16]


def test_channel_flag_reasons_of_a_channel():
    flags = np.array([0, ChannelFlag.LOW_SNR | ChannelFlag.AT_EDGE, 4])

    np.testing.assert_array_equal(
        (flags & ChannelFlag.AT_EDGE) != 0, [False, True, False]
    )
    assert list(ChannelFlag(int(flags[1]))) == [
        ChannelFlag.LOW_SNR,
        ChannelFlag.AT_EDGE,
    ]
