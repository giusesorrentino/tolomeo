from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from .._typing import Solver
from .geometry import MapGeometry

__all__ = ["BinnedMapResult", "MapResult"]


class MapResult(BaseModel):
    """
    Output of :func:`tolomeo.mapmaking.gls_map`.

    Attributes
    ----------
    image : :class:`~numpy.ndarray` of float64
        The map, in the units of the TOD. Pixel ``(iy, ix)`` is the element
        ``image[iy, ix]``; the flat kernel pixel is ``iy * nx + ix``.
        Shape ``(ny, nx)``.
    hits : :class:`~numpy.ndarray` of int64
        Number of valid samples that fell in every pixel.
        Shape ``(ny, nx)``.
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
        Noise weight of every pixel: the sum, over the samples used in the pixel, of the
        zero-lag weight of their detector (``lag_zero``). It is the diagonal of the
        preconditioner of the solver: for white noise it is proportional to
        the inverse of the variance of the pixel, for correlated noise only an
        approximation of it. 0 where ``hits == 0``. None if it was not computed.
        Shape ``(ny, nx)``.
    converged : bool
        True if the solver reached ``tol``; False if it stopped before, at ``maxiter``
        or for numerical reasons. Default ``True``.
    unit : str, optional
        Unit of ``image``, as a FITS unit string (``"K"`` for a calibrated scan). None
        means not known: the raw units of the instrument.
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


class BinnedMapResult(BaseModel):
    """
    Output of :func:`tolomeo.mapmaking.binned_map`.

    Attributes
    ----------
    image : :class:`~numpy.ndarray` of float64
        Mean of the samples in every pixel, in the units of the TOD; NaN where
        ``hits == 0``. Pixel ``(iy, ix)`` is ``image[iy, ix]``.
        Shape ``(ny, nx)``.
    hits : :class:`~numpy.ndarray` of int64
        Number of samples used in every pixel.
        Shape ``(ny, nx)``.
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
