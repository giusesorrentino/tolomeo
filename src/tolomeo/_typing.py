from typing import Literal

Projection = Literal["sin", "tan", "zea"]
Frame = Literal["radec", "azel"]
PlotAxes = Literal["offset", "radec", "pixel"]
CommonModeMethod = Literal["median", "mean"]
ResidualNorm = Literal["plain", "preconditioned"]
