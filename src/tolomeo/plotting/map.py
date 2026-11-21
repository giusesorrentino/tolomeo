import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .._typing import PlotAxes
from ..map import BinnedMapResult, MapResult

__all__ = ["plot_map"]


STYLE_FILE = Path(__file__).resolve().parent / "resources" / "default.mplstyle"
CMB_RGB_FILE = (
    Path(__file__).resolve().parent / "resources" / "Planck_Parchment_RGB.txt"
)

# levels of the contours of `plot_map`, as fractions of the peak of the pixels shown
_CONTOUR_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)

# characters that LaTeX treats as special, escaped in a unit that astropy cannot read
_LATEX_SPECIAL = re.compile(r"([_%&#$])")


@lru_cache(maxsize=1)
def _cmb_cmap() -> ListedColormap:
    """Planck "parchment" colormap of the CMB maps, read once from ``resources``."""
    return ListedColormap(np.loadtxt(CMB_RGB_FILE) / 255.0, name="cmb")


def _get_cmap(name: str) -> Colormap:
    """``"cmb"`` is the Planck colormap; any other name is a matplotlib colormap."""
    return _cmb_cmap() if name == "cmb" else plt.get_cmap(name)


def _colorbar_label(unit: str | None) -> str:
    r"""
    Label of the colorbar of a map in ``unit``.

    Parameters
    ----------
    unit : str or None
        Unit of the map, a FITS unit string as in :class:`~tolomeo.map.MapResult`;
        None if it is not known.

    Returns
    -------
    str
        "Temperature" for a temperature (Rayleigh-Jeans, as in
        :meth:`~tolomeo.map.MapResult.to`), "Surface brightness" for a flux density per
        beam and "Signal" for any other unit, followed by the unit in LaTeX, for
        example ``"Temperature [$\mathrm{mK}_\mathrm{RJ}$]"``. Without a unit it is
        ``"Signal [arbitrary units]"``; a unit that astropy cannot read is shown as it
        is.
    """
    if unit is None:
        return "Signal [arbitrary units]"
    try:
        parsed = u.Unit(unit, format="fits")
    except ValueError:
        escaped = _LATEX_SPECIAL.sub(r"\\\1", unit)
        return f"Signal [{escaped}]"

    # astropy writes the unit in math mode, for example "$\mathrm{mJy\,beam^{-1}}$"
    symbol = parsed.to_string("latex_inline").strip("$")
    if parsed.physical_type == "temperature":
        return rf"Temperature [${symbol}_\mathrm{{RJ}}$]"
    if parsed.is_equivalent(u.Jy / u.beam):
        return rf"Surface brightness [${symbol}$]"
    return rf"Signal [${symbol}$]"


def plot_map(
    result: MapResult | BinnedMapResult,
    percentiles: tuple[float, float] | None = None,
    min_hits_frac: float = 0.1,
    vmin: float | None = None,
    vmax: float | None = None,
    axes: PlotAxes = "offset",
    cmap: str = "afmhot",
    label: str | None = None,
    offset: Literal["median"] | float | None = "median",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    contour: bool = False,
) -> Figure:
    """ci penso dopo al docstring"""
    if axes not in ("offset", "radec", "pixel"):
        raise ValueError(f"`axes` must be 'offset' or 'radec', got {axes!r}.")

    if label is None:
        # a BinnedMapResult has no unit
        label = _colorbar_label(result.unit if isinstance(result, MapResult) else None)

    data = result.image
    hits = result.hits
    if not 0.0 <= min_hits_frac <= 1.0:
        raise ValueError(
            f"`min_hits_frac` must be between 0 and 1, got {min_hits_frac}."
        )
    # pixels shown: observed, with at least `min_hits_frac` of the largest number of hits
    # (the edges of the scan, with few samples, are very noisy)
    good = (hits > 0) & (hits >= min_hits_frac * hits.max())

    if not np.isfinite(data[good]).any():
        raise ValueError(" (ci penso poi all'errore) ")

    data = np.where(good, data, np.nan)

    # the zero of the map is not fixed by the GLS (a constant added to the whole map does
    # not change the cleaned TOD): set it to the median of the observed pixels, or to a
    # given value
    if offset == "median":
        data = data - np.nanmedian(data[good])
    elif offset is not None:
        data = data - offset

    geometry = result.geometry

    # create the figure before the style context: the first pyplot figure of a
    # Jupyter session turns interactive mode on, and the context would switch it
    # off again on exit, so later figures would no longer be shown
    fig = plt.figure(figsize=(6.5, 6.5))

    with plt.style.context(STYLE_FILE):
        if axes == "radec":
            # WCSAxes: the image is drawn in pixels, the axes show RA and Dec
            ax = fig.add_subplot(projection=geometry.to_wcs())
            extent = None
        elif axes == "offset":
            # offsets from the projection center, in arcmin; extent takes the outer
            # pixel edges, and the pixelization spans -n * step / 2 to +n * step / 2
            ax = fig.add_subplot()
            half_x = geometry.nx * geometry.pixel_size / 2 / 60
            half_y = geometry.ny * geometry.pixel_size / 2 / 60
            extent = (-half_x, half_x, -half_y, half_y)
        elif axes == "pixel":
            raise NotImplementedError()

        colormap = _get_cmap(cmap).with_extremes(bad="lightgray")

        # None lets imshow use the minimum and maximum of the observed pixels
        lo = hi = None
        if percentiles is not None and (vmin is None or vmax is None):
            lo, hi = np.nanpercentile(data[good], percentiles)

        if vmax is not None:
            hi = vmax
        if vmin is not None:
            lo = vmin

        # the cmb colormap is diverging: unless both limits are given, center it on zero
        # (symmetric around the one given, or around the largest of |lo| and |hi|)
        if cmap == "cmb" and (vmin is None or vmax is None):
            if vmax is not None:
                half = abs(vmax)
            elif vmin is not None:
                half = abs(vmin)
            else:
                lo = np.nanmin(data[good]) if lo is None else lo
                hi = np.nanmax(data[good]) if hi is None else hi
                half = max(abs(lo), abs(hi))
            lo, hi = -half, half

        im = ax.imshow(data, cmap=colormap, extent=extent, vmax=hi, vmin=lo)

        # contours at fractions of the peak of the pixels shown (none if the peak is not
        # positive). Same extent and origin as imshow, so the lines sit on the pixels;
        # with "radec" both are drawn in pixel coordinates
        if contour:
            peak = np.nanmax(data[good])
            if peak > 0:
                ax.contour(
                    np.ma.masked_invalid(data),
                    levels=peak * np.asarray(_CONTOUR_FRACTIONS),
                    colors="k",
                    linewidths=0.6,
                    extent=extent,
                    origin="image",
                )

        if axes == "radec":
            ax.coords[0].set_axislabel("RA")
            ax.coords[1].set_axislabel("Dec")
            # astropy picks the label syntax from text.usetex at draw time, which is
            # after the style context has closed; these separators work either way
            ax.coords[0].set_separator(
                (r"$^\mathrm{h}$", r"$^\mathrm{m}$", r"$^\mathrm{s}$")
            )
            ax.coords[1].set_separator((r"$^\circ$", r"$'$", r"$''$"))
        else:
            ax.set_xlabel(r"$\Delta x$ [arcmin]")
            ax.set_ylabel(r"$\Delta y$ [arcmin]")

        # limits in the units of the axes: arcmin for "offset", pixels for "radec" (WCSAxes
        # draws the image in pixel coordinates)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        # without axes_class the colorbar axis would be a WCSAxes too
        cax = make_axes_locatable(ax).append_axes(
            "right", size="5%", pad=0.1, axes_class=Axes
        )
        cb = fig.colorbar(im, cax=cax, label=label)
        cb.ax.tick_params(direction="out")

        return fig


def angular_spectrum():
    pass
