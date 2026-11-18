# Smoke test of the installed package: `spin smoke` runs this file as a script, with
# the wheel and then the sdist installed in an isolated environment. It does not
# import `tests.common`: run as a script, only the folder of this file is on the path.

from importlib.resources import files
from pathlib import Path

import numpy as np

import tolomeo
from tolomeo._cli import main
from tolomeo.coords import Coordinates
from tolomeo.map import MapGeometry
from tolomeo.mapmaking import binned_map, estimate_inverse_noise, gls_map
from tolomeo.plotting.map import CMB_RGB_FILE, STYLE_FILE, _cmb_cmap
from tolomeo.processing import FilteringConfig, process_tod
from tolomeo.tod import Scan, TimeOrderedData

# -----------------------------------------------------------------------------
# Installed package
# -----------------------------------------------------------------------------

src_init = Path(__file__).parents[1] / "src" / "tolomeo" / "__init__.py"
assert Path(tolomeo.__file__).resolve() != src_init.resolve()


# -----------------------------------------------------------------------------
# Pipeline on a synthetic scan
# -----------------------------------------------------------------------------

rng = np.random.default_rng(0)

arcsec = np.deg2rad(1 / 3600)
ra0, dec0 = np.deg2rad(52.0), np.deg2rad(41.5)
n_subscans, subscan_len, n_det = 4, 512, 3
n_samples = n_subscans * subscan_len

# raster: subscans along RA, one row of Dec each
along = np.linspace(-60, 60, subscan_len) * arcsec
ra = np.concatenate([ra0 + along / np.cos(dec0)] * n_subscans)
dec = np.repeat(dec0 + (np.arange(n_subscans) - 1.5) * 10 * arcsec, subscan_len)

scan = Scan(
    tod=TimeOrderedData(
        signal=rng.standard_normal((n_samples, n_det)),
        time=60600.0 + np.arange(n_samples) / 244.0 / 86400,
        flags=np.ones(n_samples, dtype=bool),
        segments=np.arange(n_subscans + 1) * subscan_len,
    ),
    coordinates=Coordinates(
        ra=ra,
        dec=dec,
        par_angle=np.zeros(n_samples),
        az=np.zeros(n_samples),
        el=np.full(n_samples, 0.8),
    ),
    offsets={"lon": np.array([-5.0, 0.0, 5.0]) * arcsec, "lat": np.zeros(n_det)},
    channel_ids=[1, 2, 3],
)

geometry = MapGeometry.from_scan(scan, pixel_size=10.0)
clean = process_tod(scan)
inv_noise = estimate_inverse_noise(clean, nperseg=128, workers=1)
result = gls_map(clean, geometry, inv_noise, tol=0.5, workers=1)
binned = binned_map(clean, geometry)

assert result.image.shape == binned.image.shape == (geometry.ny, geometry.nx)


# -----------------------------------------------------------------------------
# Package resources
# -----------------------------------------------------------------------------

config = FilteringConfig.from_yaml(
    files("tolomeo.processing") / "resources" / "filtering.yaml"
)
assert len(config.steps) > 0

assert STYLE_FILE.is_file()
assert CMB_RGB_FILE.is_file()
assert _cmb_cmap().N > 0


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------

try:
    main(["--help"])
except SystemExit as exc:
    assert exc.code == 0
