import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from tests.common import DETECTOR_IDS, SAMPLE_RATE, make_scan
from tolomeo.plotting import plot_psd

# -----------------------------------------------------------------------------
# plot_psd
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("data", ["scan", "tod"])
@pytest.mark.parametrize("mode", ["welch", "fft"])
def test_plot_psd_returns_figure(data, mode):
    # Check that both a Scan and its TimeOrderedData are accepted, in both modes.
    scan = make_scan()

    fig = plot_psd(scan if data == "scan" else scan.tod, mode=mode)

    assert isinstance(fig, Figure)
    assert len(fig.axes[0].get_lines()) == 1
    assert fig.axes[0].get_xscale() == fig.axes[0].get_yscale() == "log"

    plt.close(fig)


@pytest.mark.parametrize("mode", ["welch", "fft"])
def test_plot_psd_white_noise_level(mode):
    # White noise of rms 1 has the one-sided PSD 2 / sample_rate at every frequency.
    scan = make_scan(n_subscans=16, noise_rms=1.0)

    fig = plot_psd(scan, mode=mode)
    psd = fig.axes[0].get_lines()[0].get_ydata()

    assert np.mean(psd) == pytest.approx(2 / SAMPLE_RATE, rel=0.1)

    plt.close(fig)


@pytest.mark.parametrize("channel", ["all", 2])
def test_plot_psd_both_draws_welch_on_the_periodogram(channel):
    fig = plot_psd(make_scan(), channel=channel, mode="both")

    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert labels[0].startswith("periodogram")
    assert labels[1].startswith("Welch")
    assert len(labels) == 2

    plt.close(fig)


def test_plot_psd_one_channel_title():
    scan = make_scan()

    fig_scan = plot_psd(scan, channel=1)
    fig_tod = plot_psd(scan.tod, channel=-1)

    assert fig_scan.axes[0].get_title() == f"Channel 1 (ID {DETECTOR_IDS[1]})"
    assert fig_tod.axes[0].get_title() == f"Channel {len(DETECTOR_IDS) - 1}"

    plt.close("all")


@pytest.mark.parametrize("mode", ["welch", "fft"])
def test_plot_psd_all_channels_one_line(mode):
    # All the channels: their mean, one line with no band and no title.
    fig = plot_psd(make_scan(), channel="all", mode=mode)

    assert len(fig.axes[0].get_lines()) == 1
    assert len(fig.axes[0].collections) == 0
    assert fig.axes[0].get_title() == ""

    plt.close(fig)


def test_plot_psd_welch_starts_at_its_first_bin():
    # Nothing is drawn below sample_rate / nperseg: there Welch has no bin. The
    # sample rate of the TOD comes from times in MJD, exact to about 1e-4
    fig = plot_psd(make_scan(), nperseg=128)

    freqs = fig.axes[0].get_lines()[0].get_xdata()
    assert freqs.min() == pytest.approx(SAMPLE_RATE / 128, rel=1e-3)

    plt.close(fig)


def test_plot_psd_fft_down_to_one_over_t():
    # The periodogram of the whole chunk: its first bin is 1 / T. The sample rate of
    # the TOD comes from times in MJD, exact to about 1e-4
    scan = make_scan()

    fig = plot_psd(scan, mode="fft")

    freqs = fig.axes[0].get_lines()[0].get_xdata()
    assert freqs.min() == pytest.approx(SAMPLE_RATE / scan.tod.n_samples, rel=1e-3)

    plt.close(fig)


def test_plot_psd_fft_two_chunks_white_noise_level():
    # Two chunks of different lengths, the shorter zero padded: the level of white
    # noise of rms 1 is still 2 / sample_rate.
    scan = make_scan(n_subscans=16, noise_rms=1.0)
    tod = scan.tod.model_copy(
        update={"chunks": np.array([0, 1000, scan.tod.n_samples])}
    )

    fig = plot_psd(tod, mode="fft")
    psd = fig.axes[0].get_lines()[0].get_ydata()

    assert np.mean(psd) == pytest.approx(2 / SAMPLE_RATE, rel=0.1)

    plt.close(fig)


def test_plot_psd_fft_warns_on_jumps_in_time():
    # A TOD not patched, with a gap in time after every subscan.
    scan = make_scan(subscan_len=256, gap=1.0)

    with pytest.warns(UserWarning, match=r"jumps in time inside a chunk"):
        fig = plot_psd(scan, mode="fft")

    plt.close(fig)


def test_plot_psd_no_long_enough_stretch_raises():
    # Blocks longer than every subscan between two gaps in time.
    scan = make_scan(subscan_len=256, gap=1.0)

    with pytest.raises(ValueError, match=r"No stretch of the TOD"):
        plot_psd(scan, nperseg=512)

    plt.close("all")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"channel": "some"}, r"`channel` must be an integer or 'all'"),
        ({"channel": 99}, r"`channel` must be between 0 and"),
        ({"mode": "lomb"}, r"`mode` must be 'welch', 'fft' or 'both'"),
        ({"nperseg": 1}, r"`nperseg` must be at least 2"),
    ],
)
def test_plot_psd_wrong_arguments_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        plot_psd(make_scan(), **kwargs)

    plt.close("all")


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
