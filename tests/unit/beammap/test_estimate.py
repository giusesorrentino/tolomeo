import numpy as np
import pytest

from tests.common import (
    ARCSEC,
    OFFSETS_LAT,
    OFFSETS_LON,
    SOURCE_NAME,
    gaussian_source,
    make_scan,
    with_signal,
)
from tolomeo.beammap import ChannelFlag, reconstruct_focal_plane
from tolomeo.io import read_beammap

REF = 12  # reference channel, the third detector, at the boresight
FWHM = 30.0  # arcsec, of the simulated beams
OPTIONS = {"pixel_size": 5, "fit_radius": 40, "fwhm": FWHM}


def _beammap_scan(lon=OFFSETS_LON, lat=OFFSETS_LAT, gains=(-1.0, -1.0, -1.0, -1.0)):
    """
    Scan on a point source, as read without a beammap (zero offsets).

    The detectors are at ``lon``, ``lat`` (radians) and see the source with
    ``gains``, negative as in the raw phase of the KIDs, plus a constant and a little
    white noise.
    """
    scan = make_scan(n_subscans=15, subscan_len=1500, noise_rms=0.0, par_angle=0.4)
    offsets = {"lon": np.asarray(lon), "lat": np.asarray(lat)}
    true = scan.model_copy(update={"offsets": offsets})
    signal = gaussian_source(true, amplitude=1.0, fwhm=FWHM * ARCSEC) * gains
    signal += 5.0 + 0.02 * np.random.default_rng(0).standard_normal(signal.shape)

    n_det = len(scan.channel_ids)
    zero = {"lon": np.zeros(n_det), "lat": np.zeros(n_det)}
    return with_signal(true, signal).model_copy(update={"offsets": zero})


# -----------------------------------------------------------------------------
# reconstruct_focal_plane
# -----------------------------------------------------------------------------


def test_reconstruct_focal_plane_recovers_the_offsets():
    # Check the offsets relative to the reference, their errors, the reference, and
    # what the scan was.
    scan = _beammap_scan()

    beammap = reconstruct_focal_plane(scan, REF, **OPTIONS)

    np.testing.assert_array_equal(beammap.flags, 0)
    np.testing.assert_allclose(beammap.lon, OFFSETS_LON, atol=0.5 * ARCSEC)
    np.testing.assert_allclose(beammap.lat, OFFSETS_LAT, atol=0.5 * ARCSEC)
    assert np.all((beammap.lon_error > 0) & (beammap.lon_error < 0.5 * ARCSEC))
    assert np.all((beammap.lat_error > 0) & (beammap.lat_error < 0.5 * ARCSEC))
    assert beammap.ref_channel == REF
    assert abs(beammap.ref_lon) < 0.5 * ARCSEC  # the reference is at the boresight
    assert abs(beammap.ref_lat) < 0.5 * ARCSEC
    assert beammap.snr_threshold == 10
    assert beammap.source_name == SOURCE_NAME
    assert beammap.time_range == (scan.tod.time.min(), scan.tod.time.max())


def test_reconstruct_focal_plane_sign_of_every_channel():
    # Check that a channel that sees the source with the opposite sign is found.
    scan = _beammap_scan(gains=(-1.0, -1.0, -1.0, 1.0))

    beammap = reconstruct_focal_plane(scan, REF, **OPTIONS)

    np.testing.assert_array_equal(beammap.flags, 0)
    np.testing.assert_allclose(beammap.lon, OFFSETS_LON, atol=0.5 * ARCSEC)
    np.testing.assert_allclose(beammap.lat, OFFSETS_LAT, atol=0.5 * ARCSEC)


def test_reconstruct_focal_plane_low_snr():
    scan = _beammap_scan(gains=(-1.0, -0.002, -1.0, -1.0))

    beammap = reconstruct_focal_plane(scan, REF, **OPTIONS)

    assert beammap.flags[1] & ChannelFlag.LOW_SNR
    np.testing.assert_array_equal(beammap.flags[[0, 2, 3]], 0)


def test_reconstruct_focal_plane_at_edge():
    # Check that a detector whose source is near the edge of the scan is flagged.
    lon = OFFSETS_LON.copy()
    lon[3] = 110 * ARCSEC

    beammap = reconstruct_focal_plane(_beammap_scan(lon=lon), REF, **OPTIONS)

    assert beammap.flags[3] & ChannelFlag.AT_EDGE
    np.testing.assert_array_equal(beammap.flags[:3], 0)


def test_reconstruct_focal_plane_duplicates():
    # Check that two channels with the same data are both flagged.
    scan = _beammap_scan()
    signal = scan.tod.signal.copy()
    signal[:, 3] = signal[:, 0]

    beammap = reconstruct_focal_plane(with_signal(scan, signal), REF, **OPTIONS)

    np.testing.assert_array_equal(
        beammap.flags, [ChannelFlag.DUPLICATE, 0, 0, ChannelFlag.DUPLICATE]
    )


def test_reconstruct_focal_plane_reference_without_fit_raises():
    scan = _beammap_scan()
    signal = scan.tod.signal.copy()
    signal[:, 2] = np.nan

    with pytest.raises(ValueError, match=r"reference channel 12 has no fit"):
        reconstruct_focal_plane(with_signal(scan, signal), REF, **OPTIONS)


def test_reconstruct_focal_plane_same_result_with_threads():
    # Check that the fits in parallel threads give the same beammap as one thread.
    scan = _beammap_scan(gains=(-1.0, -0.002, -1.0, -1.0))  # one channel LOW_SNR

    serial = reconstruct_focal_plane(scan, REF, workers=None, **OPTIONS)
    parallel = reconstruct_focal_plane(scan, REF, workers=4, **OPTIONS)

    for name in ["flags", "lon", "lat", "lon_error", "lat_error"]:
        np.testing.assert_array_equal(getattr(parallel, name), getattr(serial, name))


def test_reconstruct_focal_plane_writes_the_output(tmp_path):
    path = tmp_path / "beammap.dat"

    beammap = reconstruct_focal_plane(_beammap_scan(), REF, **OPTIONS, output=path)

    read = read_beammap(path)
    np.testing.assert_array_equal(read.channel_ids, beammap.channel_ids)
    np.testing.assert_array_equal(read.flags, beammap.flags)
    np.testing.assert_allclose(read.lon, beammap.lon, atol=np.deg2rad(1e-6))
    assert read.ref_channel == REF
