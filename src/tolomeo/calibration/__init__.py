"""
Calibration of the TOD: from the units of the instrument to a temperature.

The responsivity of every detector comes from a calibration (for now the skydips,
:func:`skydip_responsivity`); :func:`calibrate` divides the cleaned TOD by it.
"""

from ._calibrate import calibrate
from ._skydip import skydip_responsivity

__all__ = ["calibrate", "skydip_responsivity"]
