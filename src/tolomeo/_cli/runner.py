import logging
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from .._lib.messages import _warn
from ..calibration import calibrate, skydip_responsivity
from ..io import read_skydip, write_map
from ..map import MapGeometry, MapResult
from ..mapmaking import InverseNoise, estimate_inverse_noise, gls_map
from ..mapmaking._noise import LeaveOneOutError
from ..plotting import plot_map
from ..processing import process_tod
from ..tod import Scan
from .config import RunConfig

logger = logging.getLogger(__name__)


class RunMapmaker:
    """
    Run the whole pipeline of a ``RunConfig``.

    ``RunConfig.from_yaml`` has already read the scan (``read_scan``) and built the
    geometry (``MapGeometry``), so they are ``config.scan`` and ``config.geometry``. The
    steps of the run are methods, one per stage of ``process_tod ->
    estimate_inverse_noise -> gls_map -> write_map``, so that each one can be called (and
    tested) alone.
    """

    def __init__(self, config: RunConfig) -> None:
        self.config = config

    def run(self) -> MapResult:
        """
        Clean and map ``config.scan``, save the results and return the map.

        Call it once: the cleaning is done in place on ``config.scan``. If the output
        file exists and ``output.overwrite`` is False it stops at once, before the
        slow part.

        Returns
        -------
        :class:`~tolomeo.map.MapResult`
            The map that was written to ``config.output.path``.
        """
        out = self.config.output
        if not out.overwrite and out.path.exists():
            raise FileExistsError(
                f"{out.path} already exists: set `output.overwrite: true` to replace it."
            )

        scan = self._clean(self.config.scan, self.config.geometry)
        scan = self._calibrate(scan)
        inv_noise = self._estimate_noise(scan)
        # after the noise estimate: NPERSEG is the value used, also when it is automatic
        header = self._header(self.config.scan, inv_noise)
        result = self._make_map(scan, self.config.geometry, inv_noise)
        self._save(result, header)

        return result

    def _header(self, scan: Scan, inv_noise: InverseNoise) -> dict:
        """
        FITS keywords of the run, from ``scan``, ``inv_noise`` and the config.

        ``TELESCOP`` and ``INSTRUME`` (``scan.telescope`` and ``scan.instrument``, if not
        None), ``DATE-OBS`` and ``MJD-OBS`` (first sample), ``NDET``, ``CALIB`` and, with a
        calibration, ``TATM`` and ``TAUATM``, then ``NPERSEG`` (from ``inv_noise``, so
        it is the value used also when ``noise.nperseg`` is ``auto``), ``GLSTOL``,
        ``MAXITER`` and ``MINHITS``. The keywords of ``config.output.header`` come last,
        so they replace these if they have the same name.
        """
        config = self.config
        mjd = float(np.nanmin(scan.tod.time))

        header = {}
        if scan.telescope is not None:
            header["TELESCOP"] = (scan.telescope, "Telescope")
        if scan.instrument is not None:
            header["INSTRUME"] = (scan.instrument, "Instrument")

        header |= {
            "DATE-OBS": (Time(mjd, format="mjd").isot, "UTC of the first sample"),
            "MJD-OBS": (mjd, "MJD of the first sample"),
            "NDET": (scan.tod.n_det, "Detectors read from the beammap"),
            "CALIB": (config.calibration is not None, "Skydip calibration applied"),
        }
        if config.calibration is not None:
            header["TATM"] = (config.calibration.T_atm, "[K] Atmosphere temperature")
            header["TAUATM"] = (config.calibration.tau_atm, "Zenith opacity")
            header["OPACORR"] = (
                config.calibration.correct_opacity,
                "Divided by the transmission",
            )

        header["NPERSEG"] = (inv_noise.nperseg, "Samples per noise block")
        header["GLSTOL"] = (config.gls.tol, "Tolerance of the solver")
        header["MAXITER"] = (config.gls.maxiter, "Maximum iterations of the solver")
        if config.gls.min_hits_frac is not None:
            header["MINHITS"] = (config.gls.min_hits_frac, "Min hits / max hits")

        header.update(config.output.header)
        return header

    def _calibrate(self, scan: Scan) -> Scan:
        """
        Calibrate ``scan`` with the skydip of ``config.calibration``.

        ``read_skydip``, then ``skydip_responsivity`` and ``calibrate``. Without a
        ``calibration`` section it returns ``scan`` itself; otherwise a new scan, and
        ``scan`` keeps the signal before the calibration.
        """
        calibration = self.config.calibration

        if calibration is None:
            return scan

        t0 = time.perf_counter()
        logger.info("[CALIBRATION] Starting the skydip calibration")

        skydips = read_skydip(calibration.skydip_dir, scan.channel_ids)
        responsivity = skydip_responsivity(
            skydips,
            T_atm=calibration.T_atm,
            tau_atm=calibration.tau_atm,
        )

        t1 = time.perf_counter()
        logger.info(f"[CALIBRATION] Responsivity computed in {t1 - t0:.2f}s")

        tau_atm = calibration.tau_atm if calibration.correct_opacity else None
        scan = calibrate(scan, responsivity, tau_atm=tau_atm)

        logger.info(f"[CALIBRATION] TOD calibrated in {time.perf_counter() - t1:.2f}s")
        logger.info(f"[CALIBRATION] Completed in {time.perf_counter() - t0:.2f}s")

        return scan

    def _clean(self, scan: Scan, geometry: MapGeometry) -> Scan:
        """
        ``process_tod(scan, config.filtering, geometry=geometry, overwrite_scan=True)``.

        ``geometry`` is needed only by a ``RadiusMask``. With ``overwrite_scan=True`` the
        signal of ``scan`` is cleaned in place and the returned scan has the same arrays.
        """
        t0 = time.perf_counter()
        logger.info("[PROCESSING] Starting to process the TOD")

        scan = process_tod(
            scan=scan,
            config=self.config.filtering,
            background=None,
            overwrite_scan=True,
            geometry=geometry,
        )

        logger.info(
            f"[PROCESSING] Cleaning completed in {time.perf_counter() - t0:.2f}s"
        )

        return scan

    def _estimate_noise(self, scan: Scan) -> InverseNoise:
        """
        ``estimate_inverse_noise(scan, ...)`` with the fields of ``config.noise``.

        ``nperseg``, ``filters``, ``mask`` and ``leave_one_out`` map one to one onto
        its arguments (same names; ``nperseg`` None is the automatic choice). Without
        ``noise.mask`` the mask of ``config.filtering`` is used, so that the source
        stays out of the spectra, unless ``noise.leave_one_out`` removes it without a
        mask. When the leave-one-out cannot be done (a single scan), a warning says so
        and the spectra are estimated on all the good samples, without a mask. The
        threads are ``config.gls.workers``.
        """
        noise = self.config.noise
        if noise.leave_one_out:
            mask = None  # NoiseConfig refuses noise.mask with the leave-one-out
        else:
            mask = noise.mask if noise.mask is not None else self.config.filtering.mask
        arguments = {
            "nperseg": noise.nperseg,
            "filters": noise.filters,
            "geometry": self.config.geometry,
            "background": None,
            "cross_spectra": False,
            "workers": self.config.gls.workers,
        }
        try:
            return estimate_inverse_noise(
                scan, mask=mask, leave_one_out=noise.leave_one_out, **arguments
            )
        except LeaveOneOutError as error:
            _warn(f"{error} The noise spectra are estimated without a mask.")
            return estimate_inverse_noise(
                scan, mask=None, leave_one_out=False, **arguments
            )

    def _make_map(
        self, scan: Scan, geometry: MapGeometry, inv_noise: InverseNoise
    ) -> MapResult:
        """
        ``gls_map(scan, geometry, inv_noise, ...)`` with the fields of ``config.gls``.

        ``solver``, ``residual_norm``, ``tol``, ``maxiter``, ``min_hits_frac`` and
        ``workers`` map one to one onto the arguments of ``gls_map`` (same names).
        """
        return gls_map(
            scan=scan,
            geometry=geometry,
            inv_noise=inv_noise,
            initial_map=None,
            solver=self.config.gls.solver,
            residual_norm=self.config.gls.residual_norm,
            tol=self.config.gls.tol,
            maxiter=self.config.gls.maxiter,
            min_hits_frac=self.config.gls.min_hits_frac,
            workers=self.config.gls.workers,
        )

    def _save(self, result: MapResult, header: dict) -> None:
        """
        Write the FITS map and, if asked, the image.

        1. Create the folder of ``config.output.path`` if it does not exist
           (``mkdir(parents=True, exist_ok=True)``).
        2. ``write_map(result, path, extra_header=header, config=config.yaml_text,
           overwrite=config.output.overwrite)``, with ``header`` from ``_header``.
        3. If ``config.output.plot`` is set: ``plot_map(result)``, ``fig.savefig(...)``
           and ``plt.close(fig)`` so the figure does not stay in memory. The command
           line may run on a machine without a display: select the ``"Agg"`` backend
           (``matplotlib.use("Agg")``) before the first figure, not after.
        """
        t0 = time.perf_counter()
        output_path = Path(self.config.output.path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"[OUTPUT] Writing the map to {output_path}")

        write_map(
            result=result,
            path=output_path,
            extra_header=header,
            config=self.config.yaml_text,
            overwrite=self.config.output.overwrite,
        )

        logger.info(f"[OUTPUT] Map written in {time.perf_counter() - t0:.2f}s")

        plot_path = self.config.output.plot
        if plot_path is not None:
            plot_path.parent.mkdir(parents=True, exist_ok=True)

            matplotlib.use("Agg")  # no display needed; before the first figure
            fig = plot_map(result=result)
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            logger.info(f"[OUTPUT] Image written to {plot_path}")
