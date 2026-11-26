r"""
Gaussian noise with a given power spectrum: free realizations, and constrained ones that
fill the gaps of a TOD.

A free realization of stationary noise with ``n`` samples is drawn in Fourier space:
white noise, times the square root of the variance per frequency
:math:`S_k = (f_s/2)\,P(f_k)` of a one-sided PSD :math:`P` (:func:`rfft_variances`,
:func:`draw_noise`). A constrained realization keeps the good samples of a TOD and
fills its gaps with noise consistent with them and with the spectrum
(:func:`fill_gaps`, the method of Hoffman & Ribak 1991).

The covariance of the noise is taken as circulant, diagonal in Fourier space with
eigenvalues :math:`S_k`: the TOD is treated as periodic, so its two ends are
correlated as if they were adjacent.

References
----------
Hoffman, Y. & Ribak, E. 1991, ApJ, 380, L5.
Stompor, R. et al. 2002, Phys. Rev. D, 65, 022003.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfft, rfft
from scipy.sparse.linalg import LinearOperator, cg

from .._lib.messages import _warn


def rfft_variances(
    freqs: NDArray[np.float64],
    psd: NDArray[np.float64],
    n_samples: int,
    sample_rate: float,
) -> NDArray[np.float64]:
    r"""
    Variance per frequency of a TOD of ``n_samples`` samples with a one-sided PSD.

    The PSD estimated at ``freqs`` (for example by :func:`scipy.signal.welch`) is
    interpolated in log-log on the frequencies of ``np.fft.rfft`` of the TOD,
    :math:`f_k = k f_s / n`, and multiplied by :math:`f_s/2`. Below the lowest
    positive frequency of ``freqs`` it keeps its first positive value, and bin 0 gets
    the value of bin 1, since :math:`\log f` does not exist at :math:`f = 0`.

    Parameters
    ----------
    freqs : :class:`~numpy.ndarray` of float64
        Frequencies of the estimate, in Hz, increasing. Shape ``(n_freqs,)``.
    psd : :class:`~numpy.ndarray` of float64
        One-sided PSD at ``freqs``, in units squared per Hz, positive.
        Shape ``(n_freqs,)``.
    n_samples : int
        Length of the TOD to draw or to fill.
    sample_rate : float
        Sampling frequency, in Hz.

    Returns
    -------
    :class:`~numpy.ndarray` of float64
        :math:`S_k`, in units squared: for white noise of rms :math:`\sigma`, every
        :math:`S_k` is :math:`\sigma^2`. Shape ``(n_samples // 2 + 1,)``.
    """
    # log-log interpolation, so bin 0 of both is left out: np.interp keeps the first
    # value below freqs[1] and the last one above freqs[-1]
    fk = np.fft.rfftfreq(n_samples, 1 / sample_rate)
    pfk = np.interp(np.log(fk[1:]), np.log(freqs[1:]), np.log(psd[1:]))

    # bin 0 gets the value of bin 1
    pfk = np.insert(pfk, 0, pfk[0])
    return sample_rate * np.exp(pfk) / 2


def draw_noise(
    variances: NDArray[np.float64],
    n_samples: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    r"""
    Free realization of stationary Gaussian noise with variances ``variances``.

    ``n_samples`` white samples :math:`w_j \sim \mathcal{N}(0, 1)`, their ``rfft``
    multiplied by :math:`\sqrt{S_k}`, and the ``irfft`` of length ``n_samples``. With a
    constant :math:`S_k = S` the result is white noise of rms :math:`\sqrt{S}`.

    Parameters
    ----------
    variances : :class:`~numpy.ndarray` of float64
        :math:`S_k`, from :func:`rfft_variances`. Shape ``(n_samples // 2 + 1,)``.
    n_samples : int
        Length of the realization.
    rng : :class:`~numpy.random.Generator`
        Source of the random numbers.

    Returns
    -------
    :class:`~numpy.ndarray` of float64
        The realization. Shape ``(n_samples,)``.
    """
    # the FFTs run on one thread: the callers already work on the detectors in parallel
    wk = rng.standard_normal(n_samples)
    Wk = rfft(wk) * np.sqrt(variances)
    return irfft(Wk, n=n_samples)


def fill_gaps(
    tod: NDArray[np.float64],
    good: NDArray[np.bool_],
    variances: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    tol: float = 1e-8,
    maxiter: int | None = None,
) -> NDArray[np.float64]:
    r"""
    Constrained realization: ``tod`` on the good samples, consistent noise in the gaps.

    With :math:`\tilde{\mathbf{n}}` a free realization (:func:`draw_noise`) and
    :math:`Q = C^{-1}` the inverse covariance, applied with :func:`_apply_precision`:

    1. :math:`\boldsymbol{\delta}` is :math:`\mathbf{d} - \tilde{\mathbf{n}}` on the
       good samples and 0 in the gaps;
    2. :math:`Q_{gg}\,\mathbf{y} = -(Q\boldsymbol{\delta})_g` is solved with the
       conjugate gradient on the gaps only (:math:`Q_{gg}` is symmetric positive
       definite);
    3. the result is :math:`\mathbf{d}` on the good samples and
       :math:`\tilde{\mathbf{n}} + \mathbf{y}` in the gaps.

    Parameters
    ----------
    tod : :class:`~numpy.ndarray` of float64
        TOD of one detector, with its mean removed; the values in the gaps are not
        used. Shape ``(n_samples,)``.
    good : :class:`~numpy.ndarray` of bool
        True for the samples kept as they are. Shape ``(n_samples,)``.
    variances : :class:`~numpy.ndarray` of float64
        :math:`S_k` of the noise, from :func:`rfft_variances` with the same
        ``n_samples``; all positive. Shape ``(n_samples // 2 + 1,)``.
    rng : :class:`~numpy.random.Generator`
        Source of the random numbers of :math:`\tilde{\mathbf{n}}`.
    tol : float, optional
        Relative residual at which the conjugate gradient stops. Defaults to ``1e-8``.
    maxiter : int, optional
        Maximum iterations of the conjugate gradient. Defaults to ``None``: the
        default of :func:`scipy.sparse.linalg.cg`.

    Returns
    -------
    :class:`~numpy.ndarray` of float64
        The filled TOD, a new array (``tod`` is not modified). Shape ``(n_samples,)``.

    Warns
    -----
    RuntimeWarning
        If the conjugate gradient does not reach ``tol`` in ``maxiter`` iterations.
    """
    gaps = ~good
    n_samples = len(tod)
    if not gaps.any():
        return tod.copy()

    noise = draw_noise(variances, n_samples, rng)

    # the mismatch between the data and the free realization, zero in the gaps: then the
    # gap entries of Q delta are Q_go delta_o, and b = -Q_go delta_o
    delta = tod - noise
    delta[gaps] = 0
    qv = _apply_precision(delta, variances)
    b = -qv[gaps]

    def gap_precision(y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Q_gg y: ``y`` in the gaps of a vector of zeros, Q, and the gaps again."""
        v = np.zeros(n_samples)
        v[gaps] = np.ravel(y)
        return _apply_precision(v, variances)[gaps]

    # Q_gg is never built: the conjugate gradient only needs its product with a vector
    n_gaps = b.size
    op = LinearOperator(shape=(n_gaps, n_gaps), matvec=gap_precision, dtype=np.float64)
    y, info = cg(A=op, b=b, rtol=tol, maxiter=maxiter)
    if info > 0:
        _warn(
            f"The conjugate gradient of the gap filling did not reach rtol = {tol} in "
            f"{info} iterations: the gaps are filled with its last iterate.",
            RuntimeWarning,
        )

    out = tod.copy()
    out[gaps] = noise[gaps] + y
    return out


def _apply_precision(
    v: NDArray[np.float64], variances: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""
    :math:`Q\mathbf{v}` with :math:`Q = C^{-1}`: ``irfft(rfft(v) / variances)``.

    Parameters
    ----------
    v : :class:`~numpy.ndarray` of float64
        A vector of the length of the TOD. Shape ``(n_samples,)``.
    variances : :class:`~numpy.ndarray` of float64
        :math:`S_k`, the eigenvalues of :math:`C`. Shape ``(n_samples // 2 + 1,)``.

    Returns
    -------
    :class:`~numpy.ndarray` of float64
        Shape ``(n_samples,)``.
    """
    return irfft(rfft(v) / variances, n=len(v))
