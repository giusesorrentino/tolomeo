import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.visualization.wcsaxes import WCSAxes
from matplotlib.figure import Figure

from tests.common import DEC0, RA0
from tolomeo.map import BinnedMapResult, MapGeometry, MapResult
from tolomeo.plotting import plot_map

rng = np.random.default_rng(31)

NX, NY = 20, 16
GEOMETRY = MapGeometry(nx=NX, ny=NY, pixel_size=6.0, center=(RA0, DEC0))


def _map_result(image=None, hits=None, unit=None):
    """A MapResult with a Gaussian source on noise, every pixel observed 10 times."""
    if image is None:
        y, x = np.mgrid[:NY, :NX]
        image = 5 * np.exp(-((x - NX / 2) ** 2 + (y - NY / 2) ** 2) / 8)
        image += 0.1 * rng.standard_normal((NY, NX))
    return MapResult(
        image=image,
        hits=np.full((NY, NX), 10) if hits is None else hits,
        n_iter=5,
        residual=1e-4,
        geometry=GEOMETRY,
        solver="cg",
        unit=unit,
    )


def _image_data(fig):
    """The array drawn by ``imshow`` on the main axes."""
    return fig.axes[0].get_images()[0].get_array()


# -----------------------------------------------------------------------------
# plot_map
# -----------------------------------------------------------------------------


def test_plot_map_returns_figure():
    fig = plot_map(_map_result())

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2  # the map and its colorbar

    plt.close(fig)


def test_plot_map_binned_result():
    result = _map_result()
    binned = BinnedMapResult(
        image=result.image, hits=result.hits, std=np.ones((NY, NX)), geometry=GEOMETRY
    )

    fig = plot_map(binned)

    assert isinstance(fig, Figure)

    plt.close(fig)


@pytest.mark.parametrize(
    ("unit", "label"),
    [
        (None, "Signal [arbitrary units]"),
        ("K", r"Temperature [$\mathrm{K}_\mathrm{RJ}$]"),
        ("uK", r"Temperature [$\mathrm{\mu K}_\mathrm{RJ}$]"),
        ("mJy beam-1", r"Surface brightness [$\mathrm{mJy\,beam^{-1}}$]"),
        ("Hz", r"Signal [$\mathrm{Hz}$]"),
        ("K_RJ", r"Signal [K\_RJ]"),  # not a FITS unit: shown as it is, escaped
    ],
)
def test_plot_map_label_from_unit(unit, label):
    fig = plot_map(_map_result(unit=unit))

    assert fig.axes[1].get_ylabel() == label

    plt.close(fig)


def test_plot_map_binned_result_label():
    # A BinnedMapResult has no unit.
    result = _map_result()
    binned = BinnedMapResult(
        image=result.image, hits=result.hits, std=np.ones((NY, NX)), geometry=GEOMETRY
    )

    fig = plot_map(binned)

    assert fig.axes[1].get_ylabel() == "Signal [arbitrary units]"

    plt.close(fig)


def test_plot_map_explicit_label():
    fig = plot_map(_map_result(unit="K"), label="Flux")

    assert fig.axes[1].get_ylabel() == "Flux"

    plt.close(fig)


def test_plot_map_offset_axes_extent():
    # Check that the "offset" axes span the map in arcmin, centred on zero.
    fig = plot_map(_map_result(), axes="offset")

    half_x, half_y = NX * 6.0 / 2 / 60, NY * 6.0 / 2 / 60
    extent = fig.axes[0].get_images()[0].get_extent()
    np.testing.assert_allclose(extent, [-half_x, half_x, -half_y, half_y])

    plt.close(fig)


def test_plot_map_radec_axes():
    fig = plot_map(_map_result(), axes="radec")

    assert isinstance(fig.axes[0], WCSAxes)

    plt.close(fig)


def test_plot_map_median_offset():
    # Check that the median of the pixels shown is subtracted by default.
    fig = plot_map(_map_result())

    assert np.isclose(np.ma.median(_image_data(fig)), 0.0)

    plt.close(fig)


def test_plot_map_explicit_offset_and_limits():
    result = _map_result()

    fig = plot_map(result, offset=1.0, vmin=-2.0, vmax=3.0)

    image = fig.axes[0].get_images()[0]
    np.testing.assert_allclose(image.get_array(), result.image - 1.0)
    assert image.get_clim() == (-2.0, 3.0)

    plt.close(fig)


def test_plot_map_percentiles():
    result = _map_result()

    fig = plot_map(result, percentiles=(5, 95), offset=None)

    lo, hi = np.percentile(result.image, [5, 95])
    np.testing.assert_allclose(fig.axes[0].get_images()[0].get_clim(), (lo, hi))

    plt.close(fig)


def test_plot_map_cmb_colormap_is_symmetric():
    # The cmb colormap is diverging: its limits are centred on zero.
    fig = plot_map(_map_result(), cmap="cmb", offset=None)

    lo, hi = fig.axes[0].get_images()[0].get_clim()
    assert np.isclose(lo, -hi)

    plt.close(fig)


def test_plot_map_hides_pixels_with_few_hits():
    # Check that the pixels with fewer than min_hits_frac of the maximum hits, and the
    # pixels never observed, are not drawn.
    hits = np.full((NY, NX), 10)
    hits[0, :] = 0
    hits[1, :] = 2

    fig = plot_map(_map_result(hits=hits), min_hits_frac=0.5)

    # imshow masks the NaN pixels
    hidden = np.ma.getmaskarray(_image_data(fig))
    assert hidden[:2].all()
    assert not hidden[2:].any()

    plt.close(fig)


def test_plot_map_contours():
    fig = plot_map(_map_result(), contour=True, offset=None)

    assert len(fig.axes[0].collections) > 0

    plt.close(fig)


def test_plot_map_invalid_axes_raises():
    with pytest.raises(ValueError, match=r"`axes` must be"):
        plot_map(_map_result(), axes="galactic")


def test_plot_map_pixel_axes_not_implemented():
    with pytest.raises(NotImplementedError):
        plot_map(_map_result(), axes="pixel")

    plt.close("all")


@pytest.mark.parametrize("min_hits_frac", [-0.1, 1.5])
def test_plot_map_invalid_min_hits_frac_raises(min_hits_frac):
    with pytest.raises(ValueError, match=r"`min_hits_frac` must be between 0 and 1"):
        plot_map(_map_result(), min_hits_frac=min_hits_frac)


def test_plot_map_nothing_to_show_raises():
    with pytest.raises(ValueError):
        plot_map(_map_result(image=np.full((NY, NX), np.nan)))
