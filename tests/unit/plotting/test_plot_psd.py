import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from tests.common import make_scan
from tolomeo.plotting import plot_psd

# -----------------------------------------------------------------------------
# plot_psd
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("data", ["scan", "tod"])
def test_plot_psd_returns_figure(data):
    # Check that both a Scan and its TimeOrderedData are accepted.
    scan = make_scan()

    fig = plot_psd(scan if data == "scan" else scan.tod)

    assert isinstance(fig, Figure)
    assert len(fig.axes[0].get_lines()) == 1
    assert fig.axes[0].get_xscale() == fig.axes[0].get_yscale() == "log"

    plt.close(fig)


def test_plot_psd_unit_in_the_label():
    fig = plot_psd(make_scan(), unit="K")

    assert "K$^2$/Hz" in fig.axes[0].get_ylabel()

    plt.close(fig)


@pytest.mark.parametrize(
    ("unit", "label"), [("rad", "rad$^2$/Hz"), (None, "arbitrary")]
)
def test_plot_psd_unit_from_the_tod(unit, label):
    scan = make_scan()
    tod = scan.tod.model_copy(update={"unit": unit})

    fig = plot_psd(tod)

    assert label in fig.axes[0].get_ylabel()

    plt.close(fig)


def test_plot_psd_no_long_enough_subscan_raises():
    # Subscans of a single sample have no spectrum.
    scan = make_scan(n_subscans=4, subscan_len=1)

    with pytest.raises(ValueError, match=r"No subscan has at least 2 samples"):
        plot_psd(scan)

    plt.close("all")
