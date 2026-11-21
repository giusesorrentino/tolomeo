from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from math import ceil
from os import cpu_count
from typing import Annotated, Literal

import numpy as np
from astropy.stats import sigma_clip
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from .._lib.units import Arcmin, Degrees
from ..coords.transforms import radec2latlon
from ..map import MapGeometry
from ..tod import Scan

__all__ = ["EdgeMask", "Mask", "RadiusMask", "SigmaClipMask"]

_ARCMIN2RAD = np.pi / (180 * 60)
_BLOCK = 16384


class RadiusMask(BaseModel):
    """
    Leave the source out of the fits of the cleaning.

    The samples closer than ``radius`` to the center of the source are not used by
    :class:`~tolomeo.processing.CommonMode`, :class:`~tolomeo.processing.Baseline` and
    :class:`~tolomeo.processing.RemoveSpline` to estimate what to subtract; the
    estimate is then subtracted from every sample. The distance is measured for every
    detector, each at its own position. The window, the filters and
    :class:`~tolomeo.processing.RemoveModes` do not use the mask.

    Attributes
    ----------
    radius : float
        Radius around the source. A number is in arcmin; ``"20 arcsec"`` or an
        :class:`~astropy.units.Quantity` is converted to arcmin. It should contain the
        whole source plus a margin for the pointing error.
    center : "auto" or tuple of float
        Center of the source: ``"auto"`` is the position written in the headers of
        the FITS files (:attr:`~tolomeo.tod.Scan.source`), otherwise ``(ra, dec)``.
        Numbers are in degrees; ``"83.6 deg"`` or a quantity is converted to degrees.
        Defaults to ``"auto"``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["radius"] = "radius"
    radius: Arcmin = Field(gt=0)  # implicit unit: arcmin
    center: Literal["auto"] | tuple[Degrees, Degrees] = "auto"  # implicit unit: deg

    def build(self, scan: Scan, geometry: MapGeometry) -> NDArray[np.bool_]:
        """
        Samples that are far from the source, for every detector.

        Parameters
        ----------
        scan : :class:`~tolomeo.tod.Scan`
            Scan to clean. Its boresight (``scan.coordinates``) and the detector offsets
            (``scan.offsets``) give the position of every detector at every sample.
        geometry : :class:`~tolomeo.map.MapGeometry`
            Geometry of the map. Only its projection and frame are used: the distance
            from the center is measured on that projection, centered on ``center``.
            None is refused (``process_tod`` and ``estimate_inverse_noise`` pass None
            when they are called without ``geometry``).

        Returns
        -------
        :class:`~numpy.ndarray` of bool
            True where the detector is at least ``radius`` away from the center: the
            samples that can be used to estimate the background.
            Shape ``(n_samples, n_det)``.
        """
        if geometry is None:
            raise ValueError(
                "A radius mask needs `geometry`: the distance from the source is "
                "measured on the plane of the projection of the map."
            )
        if self.center == "auto":
            ra0, dec0 = scan.source
        else:
            ra0, dec0 = np.deg2rad(self.center)

        radius = self.radius * _ARCMIN2RAD
        coords, offsets = scan.coordinates, scan.offsets
        n_samples, n_det = coords.ra.size, offsets["lon"].size
        background = np.zeros((n_samples, n_det), dtype=bool)

        def mask_block(start: int, stop: int) -> None:
            lat, lon = radec2latlon(
                coords.ra[start:stop],
                coords.dec[start:stop],
                ra0,
                dec0,
                geometry.projection,
                coords.par_angle[start:stop],
                offsets["lon"],
                offsets["lat"],
                frame=geometry.frame,
            )

            # a boresight with NaN coordinates gives a NaN distance, and NaN >= radius is
            # False: those samples come out as not usable, which is what we want
            background[start:stop] = (np.hypot(lat, lon) >= radius).T

        workers = cpu_count() or 1
        futures = []

        block_len = min(_BLOCK, ceil(n_samples / (4 * workers)))
        chunk_edges = [*range(0, n_samples, block_len), n_samples]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for start, stop in pairwise(chunk_edges):
                futures.append(executor.submit(mask_block, start, stop))

        for future in futures:
            future.result()  # nothing to collect: it only re-raises the errors of the blocks

        return background


# The next two masks do not need to know where the source is: they are what MIRTOS calls
# "mask without radius". In MIRTOS `radius: 0` silently switches to them; here the
# kind of mask is explicit (``type``), so a wrong radius cannot change the algorithm.
# "No mask" (MIRTOS mode "none") is ``mask: None``.
class SigmaClipMask(BaseModel):
    """
    Leave the source out of the fits of the cleaning, finding it by its amplitude.

    In every subscan, the samples of a detector that are more than ``sigma`` standard
    deviations away from its median are taken to be the source, and are not used by
    :class:`~tolomeo.processing.CommonMode`, :class:`~tolomeo.processing.Baseline` and
    :class:`~tolomeo.processing.RemoveSpline` to estimate what to subtract; the
    estimate is then subtracted from every sample. Each detector has its own mask. It
    does not need the position of the source, so
    :func:`~tolomeo.processing.process_tod` does not need ``geometry`` with it.

    Attributes
    ----------
    sigma : float
        Clipping threshold, in standard deviations. Defaults to ``6``.
    maxiters : int
        Maximum number of clipping iterations. Defaults to ``10``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["sigma_clip"] = "sigma_clip"
    sigma: float = Field(default=6.0, gt=0)
    maxiters: int = Field(default=10, ge=1)

    def build(self, scan: Scan) -> NDArray[np.bool_]:
        """
        Samples that do not stand out of the noise, for every detector.

        The source is not looked for by position but by amplitude: in each subscan, the
        samples of a detector that are more than ``sigma`` standard deviations away
        from its median are taken to be the source, and left out.

        Parameters
        ----------
        scan : :class:`~tolomeo.tod.Scan`
            Scan to clean. Only ``scan.tod`` is used: the signal, the flags and the
            subscan boundaries (``segments``).

        Returns
        -------
        :class:`~numpy.ndarray` of bool
            True where the sample can be used to estimate the background: it was not
            clipped and it is not flagged. Shape ``(n_samples, n_det)``, one mask per
            detector.
        """
        tod = scan.tod
        background = np.zeros_like(tod.signal, dtype=bool)

        workers = cpu_count() or 1
        futures = []

        def mask_block(start: int, stop: int) -> None:
            signal = np.where(
                tod.flags[start:stop, None], tod.signal[start:stop], np.nan
            )
            clipped = sigma_clip(
                signal,
                sigma=self.sigma,
                maxiters=self.maxiters,
                axis=0,
                stdfunc="mad_std",
            )
            background[start:stop] = ~clipped.mask

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for start, stop in pairwise(tod.segments):
                if stop <= start:
                    continue
                futures.append(executor.submit(mask_block, start, stop))

        for future in futures:
            future.result()

        return background


class EdgeMask(BaseModel):
    """
    Leave the source out of the fits of the cleaning, assuming it is in the middle.

    Only the samples at the two ends of every subscan are used by
    :class:`~tolomeo.processing.CommonMode`, :class:`~tolomeo.processing.Baseline` and
    :class:`~tolomeo.processing.RemoveSpline` to estimate what to subtract; the
    estimate is then subtracted from every sample. The same samples are used for
    every detector, so with :class:`~tolomeo.processing.CommonMode` no detector is left
    in the middle of the subscan: it is meant for :class:`~tolomeo.processing.Baseline`
    and :class:`~tolomeo.processing.RemoveSpline`.

    Attributes
    ----------
    fraction : float
        Part of each subscan, at each end, that is used: ``0.1`` is the first 10% and
        the last 10% of the samples. Between 0 and 0.5 (excluded). Defaults to ``0.1``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["edges"] = "edges"
    fraction: float = Field(default=0.1, gt=0, lt=0.5)

    def build(self, scan: Scan) -> NDArray[np.bool_]:
        """
        Samples at the two ends of every subscan.

        It assumes that the telescope scans across the source, so that the source is
        in the middle of each subscan and the two ends are clean.

        Parameters
        ----------
        scan : :class:`~tolomeo.tod.Scan`
            Scan to clean. Only ``scan.tod`` is used: the flags and the subscan
            boundaries (``segments``).

        Returns
        -------
        :class:`~numpy.ndarray` of bool
            True for the first and the last ``fraction`` of the samples of each
            subscan, and not flagged. Shape ``(n_samples,)``: the same samples for every
            detector, which is accepted by the steps and takes far less memory than one
            mask per detector.
        """
        tod = scan.tod
        background = np.zeros(tod.signal.shape[0], dtype=bool)

        for start, stop in pairwise(tod.segments):
            if stop <= start:
                continue
            n_edge = int(self.fraction * (stop - start))
            background[start : start + n_edge] = True
            background[stop - n_edge : stop] = True
            background[start:stop] &= scan.tod.flags[start:stop]

        return background


Mask = Annotated[RadiusMask | SigmaClipMask | EdgeMask, Field(discriminator="type")]
"""Any mask of the cleaning; the ``type`` field says which one."""
