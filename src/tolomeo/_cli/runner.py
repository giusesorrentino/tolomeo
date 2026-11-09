"""``RunMapmaker``: the pipeline of the command line, from a ``RunConfig`` to a FITS map."""

import logging
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from ..io import write_map
from ..map import MapGeometry, MapResult
from ..mapmaking import gls_map
from ..plotting import plot_map
from ..processing import calibrate, process_tod, skydip_responsivity
from ..tod import Scan
from .config import RunConfig

logger = logging.getLogger("tolomeo")


class RunMapmaker:
    """
    Run the whole pipeline of a ``RunConfig``.

    ``RunConfig.from_yaml`` has already read the scan (``read_scan``) and built the
    geometry (``MapGeometry``), so they are ``config.scan`` and ``config.geometry``. The
    steps of the run are methods, one per stage of ``process_tod -> gls_map ->
    write_map``, so that each one can be called (and tested) alone.
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

        scan = self._calibrate(self.config.scan)
        scan = self._clean(scan, self.config.geometry)
        result = self._make_map(scan, self.config.geometry)
        self._save(result)

        return result

    def _calibrate(self, scan: Scan) -> Scan:
        """
        Calibrate ``scan`` with the skydip of ``config.calibration``.

        ``calibrate(scan, skydip_responsivity(...))``. Without a
        ``calibration`` section it returns ``scan`` itself; otherwise a new scan, and
        ``config.scan`` keeps the raw signal.
        """
        calibration = self.config.calibration

        if calibration is None:
            return scan

        t0 = time.perf_counter()
        print("[CALIBRATION] Starting the skydip calibration")

        responsivity = skydip_responsivity(
            calibration.skydip_dir,
            scan.detector_ids,
            T_atm=calibration.T_atm,
            tau_atm=calibration.tau_atm,
        )
        t1 = time.perf_counter()
        print(f"[CALIBRATION] Responsivity computed in {t1 - t0:.2f}s")

        tau_atm = calibration.tau_atm if calibration.correct_opacity else None
        scan = calibrate(scan, responsivity, tau_atm=tau_atm)

        print(f"[CALIBRATION] TOD calibrated in {time.perf_counter() - t1:.2f}s")
        print(f"[CALIBRATION] Completed in {time.perf_counter() - t0:.2f}s")

        return scan

    def _clean(self, scan: Scan, geometry: MapGeometry) -> Scan:
        """
        ``process_tod(scan, config.filtering, geometry=geometry, overwrite_scan=True)``.

        ``geometry`` is needed only by a ``RadiusMask``. With ``overwrite_scan=True`` the
        signal of ``scan`` is cleaned in place and the returned scan has the same arrays.
        """
        t0 = time.perf_counter()
        print("[PROCESSING] Starting to process the TOD")

        scan = process_tod(
            scan=scan,
            config=self.config.filtering,
            background=None,
            overwrite_scan=True,
            geometry=geometry,
        )

        print(f"[PROCESSING] Cleaning completed in {time.perf_counter() - t0:.2f}s")

        return scan

    def _make_map(self, scan: Scan, geometry: MapGeometry) -> MapResult:
        """
        ``gls_map(scan, geometry, ...)`` with the fields of ``config.gls``.

        ``n_freqs``, ``residual_norm``, ``tol``, ``maxiter``, ``min_hits_frac`` and
        ``workers`` map one to one onto the arguments of ``gls_map`` (same names).
        """
        return gls_map(
            scan=scan,
            geometry=geometry,
            n_freqs=self.config.gls.n_freqs,
            cross_spectra=False,
            background=None,
            initial_map=None,
            residual_norm=self.config.gls.residual_norm,
            tol=self.config.gls.tol,
            maxiter=self.config.gls.maxiter,
            min_hits_frac=self.config.gls.min_hits_frac,
            workers=self.config.gls.workers,
        )

    def _save(self, result: MapResult) -> None:
        """
        Write the FITS map and, if asked, the image.

        1. Create the folder of ``config.output.path`` if it does not exist
           (``mkdir(parents=True, exist_ok=True)``).
        2. ``write_map(result, path, extra_header=config.output.header,
           overwrite=config.output.overwrite)``.
        3. If ``config.output.plot`` is set: ``plot_map(result)``, ``fig.savefig(...)``
           and ``plt.close(fig)`` so the figure does not stay in memory. The command
           line may run on a machine without a display: select the ``"Agg"`` backend
           (``matplotlib.use("Agg")``) before the first figure, not after.
        """
        t0 = time.perf_counter()
        output_path = Path(self.config.output.path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[OUTPUT] Writing the map to {output_path}")

        write_map(
            result=result,
            path=output_path,
            extra_header=self.config.output.header,
            overwrite=self.config.output.overwrite,
        )

        print(f"[OUTPUT] Map written in {time.perf_counter() - t0:.2f}s")

        plot_path = self.config.output.plot
        if plot_path is not None:
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            
            matplotlib.use("Agg")  # no display needed; before the first figure
            fig = plot_map(result=result)
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            print(f"[OUTPUT] Image written to {plot_path}")
