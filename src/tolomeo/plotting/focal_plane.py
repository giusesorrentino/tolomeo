from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.typing import LegendLocType

from .._lib.units import _to_unit
from ..beammap import BeamMap, ChannelFlag

__all__ = ["plot_focal_plane"]

STYLE_FILE = Path(__file__).resolve().parent / "resources" / "default.mplstyle"

_RAD2ARCMIN = 60 * 180 / np.pi

# good channels in blue, flagged ones in red, a pair that stays distinct for the
# colour-blind; the reason of a flag is in the shape of its marker
_GOOD = "#2a78d6"
_BAD = "#d62728"

# label and marker of every reason; None: no position, the channels are only counted
_REASONS = {
    ChannelFlag.FIT_FAILED: ("fit failed", None),
    ChannelFlag.LOW_SNR: ("low S/N", {"marker": "x", "color": _BAD}),
    ChannelFlag.DUPLICATE: (
        "duplicate",
        {"marker": "D", "facecolors": "none", "edgecolors": _BAD},
    ),
    ChannelFlag.AT_EDGE: ("at the edge", {"marker": "^", "color": _BAD}),
    ChannelFlag.UNKNOWN: (
        "reason unknown",
        {"marker": "s", "facecolors": "none", "edgecolors": _BAD},
    ),
}


def plot_focal_plane(
    beammap: BeamMap,
    fov: float | str = 4,
    loc: LegendLocType | None = "best",
) -> Figure:
    """
    Positions of the channels of a beammap on the focal plane.

    The good channels are dots; the flagged ones get a marker per reason, on top of
    each other for a channel with more reasons. The legend counts the channels of every
    kind, and those outside the axes, which cover the field of view with a margin. A
    channel whose fit failed has no position: it is only counted.

    A beammap read from an old file only knows that a channel is bad
    (:attr:`~tolomeo.beammap.ChannelFlag.UNKNOWN`): its legend has only the good
    channels and those with an unknown reason. A new beammap always gives the
    reasons, and its legend leaves out the unknown one.

    Parameters
    ----------
    beammap : :class:`~tolomeo.beammap.BeamMap`
        Offsets and flags of the channels.
    fov : float or str, optional
        Diameter of the field of view, drawn as a dashed circle. A number is in
        arcmin. Defaults to ``4``.
    loc : str or pair of floats
        The location of the legend. The string locations place the legend at the
        corresponding position within the bounding box, which by default is
        the full axes area.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`
        Offset in azimuth on x and in elevation on y, in arcmin, as in the maps in the
        ``"azel"`` frame.
    """
    fov = _to_unit(fov, u.arcmin)
    lon = beammap.lon * _RAD2ARCMIN
    lat = beammap.lat * _RAD2ARCMIN
    flags = beammap.flags
    good = flags == 0

    # the axes cover the field of view with a margin; the channels outside are only
    # counted (NaN gives False: a failed fit is not outside)
    lim = 0.6 * fov
    outside = (np.abs(lon) > lim) | (np.abs(lat) > lim)

    with plt.style.context(STYLE_FILE):
        fig = plt.figure(figsize=(6.5, 6.5))
        ax = fig.add_subplot()

        ax.add_patch(
            plt.Circle((0, 0), fov / 2, color="0.6", fill=False, ls="--", lw=0.8)
        )
        ax.text(0.74 * fov / 2, 0.74 * fov / 2, "FOV", color="0.4", fontsize="small")

        # good channels: dots with a white ring, so that close ones stay apart
        handles = [
            ax.scatter(
                lon[good],
                lat[good],
                marker="o",
                color=_GOOD,
                zorder=2,
                label=f"good: {good.sum()}",
            )
        ]

        # flagged channels: one marker per reason. UNKNOWN is listed only if a
        # channel has it (an old file), and then the other reasons only if they
        # occur: an old file does not have them
        old_file = np.any((flags & ChannelFlag.UNKNOWN) != 0)
        for flag in ChannelFlag:
            mask = (flags & flag) != 0
            if not mask.any() and (flag is ChannelFlag.UNKNOWN or old_file):
                continue
            name = flag.name.replace("_", " ").lower()  # LaTeX: no underscores
            label, style = _REASONS.get(flag, (name, None))
            text = f"{label}: {mask.sum()}"
            if style is None:
                handles.append(Line2D([], [], ls="", label=f"{text} (no position)"))
                continue
            n_out = (mask & outside).sum()
            if n_out:
                text += f" ({n_out} outside)"
            handles.append(
                ax.scatter(
                    lon[mask],
                    lat[mask],
                    zorder=3,
                    label=text,
                    **style,
                )
            )

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\Delta\mathrm{az}\,\cos(\mathrm{el})$ [arcmin]")
        ax.set_ylabel(r"$\Delta\mathrm{el}$ [arcmin]")
        # inside the axes, in the lower right corner, outside the field of view
        if loc is not None:
            ax.legend(handles=handles, loc=loc, handletextpad=0.3)

    return fig
