"""
Flagged samples and the virtual pixel.

The mapmaker does not drop flagged samples. They still go through the noise filter, the
right-hand side, the hit count and the preconditioner, but their pixel is replaced by
one extra *virtual pixel*, so they can no longer touch any pixel of the sky map. The
virtual pixel is a normal unknown of the linear system and is thrown away at the end.

Convention (zero-based): the map pixels are ``0 .. n_pixels`` and the virtual pixel is
``n_pixels``, so the solver works on ``n_pixels + 1`` unknowns.

All the arrays here have shape ``(n_samples, n_det)``, like ``tod``.
"""

import numpy as np
from numpy.typing import NDArray


def _count_flagged(valid: NDArray[np.bool_]) -> int:
    """Number of flagged samples, i.e. the ``False`` values of ``valid``."""
    # np.count_nonzero is the fast way to count the True values
    return valid.size - np.count_nonzero(valid)


def _redirect_flagged_samples(
    pixels: NDArray[np.integer],
    valid: NDArray[np.bool_],
    virtual_pixel: int,
) -> NDArray[np.integer]:
    """
    Copy of ``pixels`` where every flagged sample points to ``virtual_pixel``.

    The result has the shape and dtype of ``pixels``. It is a *copy*: the caller's
    array is never modified, because the original pixels are still needed afterwards
    (for the hit map). Raise ``ValueError`` if the two shapes differ.
    """
    if pixels.shape != valid.shape:
        raise ValueError(
            f"`pixels` and `valid` must have the same shape, "
            f"got {pixels.shape} and {valid.shape}."
        )

    # np.where picks the pixel where the sample is valid and the virtual pixel where it
    # is not. The virtual pixel is turned into a scalar of the dtype of pixels, so that
    # the result keeps that dtype (a plain Python int has no .astype)
    return np.where(valid, pixels, pixels.dtype.type(virtual_pixel))


def _first_out_of_range_pixel(
    pixels: NDArray[np.integer],
    valid: NDArray[np.bool_] | None,
    n_pixels: int,
) -> tuple[int, int] | None:
    """
    ``(sample, det)`` of the first *valid* sample with a pixel outside the map.

    Flagged samples are ignored, because their pixel is replaced by the virtual pixel
    anyway. ``valid=None`` means that every sample counts. Return ``None`` if all the
    pixels are inside the map.
    """
    # pixel ids are unsigned (uintp), so the only way out of the map is being too large
    mask = pixels >= n_pixels

    # keep only the valid samples (in place: `mask` is a new array, not the caller's)
    if valid is not None:
        mask &= valid

    if not mask.any():
        return None

    # argmax of a boolean array is the position of the first True (in the flattened
    # array); unravel_index turns it back into (sample, det)
    idx = np.argmax(mask)
    return np.unravel_index(idx, mask.shape)
