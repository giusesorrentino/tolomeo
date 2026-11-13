"""
Diagonal of the preconditioner.

This file only *builds* ``diag``. Applying the preconditioner, ``M^-1 r``, is the
``precondition`` method of ``TemperatureNormalOperator`` in ``_operators.py``.

The preconditioner ``M`` is the diagonal of ``P^T N^-1 P`` approximated with the zero-lag
noise weight of each detector: for every sample (flagged ones included, they sit in the
virtual pixel) add ``lag_zero[det]`` to the pixel it points to. So ``diag[p]`` is the
weighted number of samples that fall in pixel ``p``.

``lag_zero`` has one value per detector; it is computed by ``estimate_inverse_noise``.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray


def _build_diagonal_preconditioner(
    pixels_by_det: NDArray[np.integer],
    lag_zero: NDArray[np.float64],
    n_unknowns: int,
    workers: int = 1,  # idk
) -> NDArray[np.float64]:
    """
    ``diag`` of length ``n_unknowns`` (map pixels + the virtual pixel).

    ``pixels_by_det`` has shape ``(n_det, n_samples)`` and dtype ``intp``: the pixel of
    every sample, one row per detector (``PointingOperator.pixels_by_det``). Detector
    major, so that the pixels of one detector are contiguous in memory and the loop
    below reads them sequentially.
    """
    diag = np.zeros(n_unknowns)
    n_det = pixels_by_det.shape[0]

    def run_det(pixels_by_det: NDArray[np.integer], idet: int) -> NDArray[np.floating]:
        """Contribution of detector ``idet`` to the diagonal, shape ``(n_unknowns,)``."""
        local_diag = np.zeros(n_unknowns)
        ids = pixels_by_det[idet]
        counts = np.bincount(ids, minlength=n_unknowns)
        local_diag += lag_zero[idet] * counts

        return local_diag

    futures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for idet in range(n_det):
            futures.append(executor.submit(run_det, pixels_by_det, idet))

    for future in futures:
        diag += future.result()

    # for idet in range(n_det):
    #     # the pixels hit by this detector, one per sample: a contiguous row
    #     ids = pixels_by_det[idet]

    #     # number of samples of this detector in each pixel. minlength makes the result
    #     # always have n_unknowns elements, also when the last pixels (the virtual one
    #     # among them) have no hit, so that it can be added to diag
    #     counts = np.bincount(ids, minlength=n_unknowns)

    #     # every sample of this detector weighs lag_zero[idet]
    #     diag += lag_zero[idet] * counts

    return diag
