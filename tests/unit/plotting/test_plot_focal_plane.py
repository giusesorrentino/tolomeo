import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from tolomeo.beammap import BeamMap, ChannelFlag
from tolomeo.plotting import plot_focal_plane

ARCMIN = np.deg2rad(1 / 60)

# -----------------------------------------------------------------------------
# plot_focal_plane
# -----------------------------------------------------------------------------


def test_plot_focal_plane_counts_every_reason():
    # Check the legend: a channel with two reasons counts for both, a failed fit
    # has no position, a channel outside the axes is counted as such, and the
    # unknown reason of the old files is left out.
    beammap = BeamMap(
        channel_ids=np.arange(5),
        flags=np.array(
            [0, 0, ChannelFlag.FIT_FAILED, ChannelFlag.LOW_SNR | ChannelFlag.AT_EDGE, 4]
        ),
        lat=np.array([0.0, 0.5, np.nan, 0.2, -1.0]) * ARCMIN,
        lon=np.array([0.0, -0.5, np.nan, 3.0, 1.0]) * ARCMIN,
    )

    fig = plot_focal_plane(beammap)

    assert isinstance(fig, Figure)
    texts = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert "good: 2" in texts
    assert "fit failed: 1 (no position)" in texts
    assert "low S/N: 1 (1 outside)" in texts
    assert "at the edge: 1 (1 outside)" in texts
    assert "duplicate: 1" in texts
    assert not any(text.startswith("reason unknown") for text in texts)

    plt.close(fig)


def test_plot_focal_plane_old_file():
    # Check that the legend of a beammap from an old file has only the good
    # channels and those with an unknown reason.
    beammap = BeamMap(
        channel_ids=np.arange(3),
        flags=np.array([0, ChannelFlag.UNKNOWN, ChannelFlag.UNKNOWN]),
        lat=np.array([0.0, 0.5, -0.5]) * ARCMIN,
        lon=np.array([0.0, -0.5, 0.5]) * ARCMIN,
    )

    fig = plot_focal_plane(beammap)

    texts = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert texts == ["good: 1", "reason unknown: 2"]

    plt.close(fig)


def test_plot_focal_plane_new_file_lists_every_reason():
    # Check that a new beammap lists the reasons that no channel has, with 0.
    beammap = BeamMap(
        channel_ids=np.arange(2),
        flags=np.array([0, ChannelFlag.LOW_SNR]),
        lat=np.array([0.0, 0.5]) * ARCMIN,
        lon=np.array([0.0, -0.5]) * ARCMIN,
    )

    fig = plot_focal_plane(beammap)

    texts = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert texts == [
        "good: 1",
        "fit failed: 0 (no position)",
        "low S/N: 1",
        "duplicate: 0",
        "at the edge: 0",
    ]

    plt.close(fig)
