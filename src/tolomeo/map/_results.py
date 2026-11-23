from typing import Any, Self

import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from .._lib.units import _conversion_factor
from .._typing import Solver
from ._geometry import MapGeometry

__all__ = ["BinnedMapResult", "MapResult"]


class MapResult(BaseModel):
    r"""
    Output of :func:`tolomeo.mapmaking.gls_map`.

    Attributes
    ----------
    image : :class:`~numpy.ndarray` of float64
        The map, in the units of the TOD. Pixel ``(iy, ix)`` is the element
        ``image[iy, ix]``; the flat kernel pixel is ``iy * nx + ix``.
        Shape ``(ny, nx)``.
    hits : :class:`~numpy.ndarray` of int64
        Number of valid samples that fell in every pixel. Shape ``(ny, nx)``.
    n_iter : int
        Iterations done by the solver.
    residual : float
        Final relative residual of the solver, in the norm ``residual_norm`` of
        :func:`~tolomeo.mapmaking.gls_map`.
    geometry : :class:`~tolomeo.map.MapGeometry`
        Geometry the map was made with: the FITS writer needs it for the WCS.
    solver : :class:`~tolomeo._typing.Solver`
        Solver that made the map: ``"cg"`` (conjugate gradient) or ``"minres"``.
    weights : :class:`~numpy.ndarray` of float64, optional
        Noise weight of every pixel, :math:`A_{pp} = \sum_j \ell_j N_{pj}`: the sum,
        over the samples used in the pixel, of the zero-lag weight :math:`\ell_j` of
        their detector (``lag_zero``), with :math:`N_{pj}` the samples of detector
        :math:`j` in pixel :math:`p`. Up to a constant factor it is the diagonal of
        :math:`A = P^T N^{-1} P`, the preconditioner of the solver. For white noise it
        is :math:`f_s/4` times the inverse of the variance of the pixel (see
        :class:`~tolomeo.mapmaking.InverseNoise`); for correlated noise the variance
        is :math:`(A^{-1})_{pp}`, which :math:`1/A_{pp}` only approximates. 0 where
        ``hits == 0``. None if it was not computed. Shape ``(ny, nx)``.
    converged : bool
        True if the solver reached ``tol``; False if it stopped before, at ``maxiter``
        or for numerical reasons. Defaults to ``True``.
    unit : str, optional
        Unit of ``image``, as a FITS unit string (``"K"`` for a calibrated scan). None
        means not known: the raw units of the instrument.
    source_name : str, optional
        Name of the observed source, ``scan.coordinates.source_name``: the ``SOURCE``
        keyword of the FITS files without the direction of the scan.
        :func:`~tolomeo.io.write_map` writes it as ``OBJECT``. None if it is not
        known. Defaults to ``None``.
    """

    # numpy arrays are not pydantic types: they are only checked with isinstance
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    image: NDArray[np.float64]
    hits: NDArray[np.int64]
    n_iter: int
    residual: float
    geometry: MapGeometry
    solver: Solver
    weights: NDArray[np.float64] | None = None
    converged: bool = True
    unit: str | None = None
    source_name: str | None = None

    def model_post_init(self, context: Any, /) -> None:
        """Raise ``ValueError`` if ``image``, ``hits`` or ``weights`` is not ``(ny, nx)``."""
        # image, hits and weights must be 2D maps with the shape of the geometry
        shape = (self.geometry.ny, self.geometry.nx)
        for name in ("image", "hits", "weights"):
            array = getattr(self, name)
            if array is None:  # only `weights` can be missing
                continue
            if array.shape != shape:
                raise ValueError(
                    f"`{name}` must have shape (ny, nx) = {shape}, got {array.shape}."
                )

    def to(
        self,
        unit: str | u.UnitBase,
        *,
        frequency: float | str | u.Quantity = 90.0,
        beam_fwhm: float | str | u.Quantity = 12.0,
    ) -> Self:
        r"""
        The same map in another unit.

        ``image`` is multiplied by the conversion factor :math:`c` and ``weights``,
        an inverse variance, is divided by :math:`c^2`; the other fields do not
        change. Between a temperature and a flux density per beam (``"Jy/beam"``) the
        conversion is the Rayleigh-Jeans one, at ``frequency`` and for a Gaussian beam
        of FWHM ``beam_fwhm`` (see Notes): they are not used for the other units.

        Parameters
        ----------
        unit : str or :class:`~astropy.units.UnitBase`
            The new unit, for example ``"mK"``, ``"uK"`` or ``"mJy/beam"``. A string is
            read as a FITS unit string.
        frequency : float, str or :class:`~astropy.units.Quantity`, optional
            Frequency of the conversion to or from ``Jy/beam``. A number is in GHz.
            Defaults to ``90``.
        beam_fwhm : float, str or :class:`~astropy.units.Quantity`, optional
            FWHM of the beam of the map, for the conversion to or from ``Jy/beam``. A
            number is in arcsec. For a smoothed map it is the FWHM of the beam
            convolved with the smoothing kernel. Defaults to ``12``.

        Returns
        -------
        :class:`~tolomeo.map.MapResult`
            A copy with the new ``image``, ``weights`` and ``unit``, the last one in
            the FITS syntax (``"Jy beam-1"`` for ``"Jy/beam"``). The other arrays are
            shared with this map, not copied.

        Raises
        ------
        ValueError
            If the unit of this map is not known (None), if ``unit`` is not a FITS
            unit, if the two units cannot be converted into each other, or if
            ``frequency`` or ``beam_fwhm`` is not positive.

        Notes
        -----
        In the Rayleigh-Jeans limit a brightness temperature :math:`T` gives the flux
        density per beam

        .. math::

            S_\nu = \frac{2 k_B \nu^2}{c^2}\, T\, \Omega_b, \qquad
            \Omega_b = \frac{\pi\, \theta_\mathrm{FWHM}^2}{4 \ln 2},

        with :math:`\Omega_b` the solid angle of a Gaussian beam: 1 K is 0.954 Jy/beam
        at 90 GHz with a FWHM of 12 arcsec. :math:`T` is the main-beam brightness
        temperature. A map calibrated on the atmosphere (skydips) may be in
        :math:`T_A^*` instead, which for a compact source is
        :math:`\eta_\mathrm{mb}\, T`: its flux densities would come out lower by the
        main-beam efficiency :math:`\eta_\mathrm{mb}`.
        """
        factor, unit = _conversion_factor(
            self.unit, unit, frequency=frequency, beam_fwhm=beam_fwhm
        )

        # new arrays: the copy shares with this map the fields that are not updated
        weights = None if self.weights is None else self.weights / factor**2
        return self.model_copy(
            update={"image": factor * self.image, "weights": weights, "unit": unit}
        )


class BinnedMapResult(BaseModel):
    """
    Output of :func:`tolomeo.mapmaking.binned_map`.

    Attributes
    ----------
    image : :class:`~numpy.ndarray` of float64
        Mean of the samples in every pixel, in the units of the TOD; NaN where
        ``hits == 0``. Pixel ``(iy, ix)`` is ``image[iy, ix]``. Shape ``(ny, nx)``.
    hits : :class:`~numpy.ndarray` of int64
        Number of samples used in every pixel. Shape ``(ny, nx)``.
    std : :class:`~numpy.ndarray` of float64
        Sample standard deviation in every pixel; NaN where ``hits <= 1``.
        Shape ``(ny, nx)``.
    geometry : :class:`~tolomeo.map.MapGeometry`
        Geometry the map was made with: the FITS writer needs it for the WCS.

    Unlike :class:`~tolomeo.map.MapResult`, the shapes are not checked.
    """

    # numpy arrays are not pydantic types: they are only checked with isinstance
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    image: NDArray[
        np.float64
    ]  # mean of the samples in every pixel, NaN where hits == 0
    hits: NDArray[np.int64]
    std: NDArray[np.float64]  # sample std in every pixel, NaN where hits <= 1
    geometry: MapGeometry

    # def __post_init__(self) -> None:
    #     # image and hits must be 2D maps with the shape of the geometry
    #     shape = (self.geometry.ny, self.geometry.nx)
    #     for name in ("image", "hits"):
    #         array = getattr(self, name)
    #         if array.shape != shape:
    #             raise ValueError(
    #                 f"`{name}` must have shape (ny, nx) = {shape}, got {array.shape}."
    #             )
