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
    # has no position, and a channel outside the axes is counted as such.
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

    plt.close(fig)
