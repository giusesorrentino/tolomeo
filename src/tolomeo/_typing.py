from typing import Literal

Projection = Literal["sin", "tan", "zea"]
Frame = Literal["radec", "azel"]
PlotCoords = Literal["offset", "world", "pixel"]
CommonModeMethod = Literal["median", "mean", "corr"]
ResidualNorm = Literal["plain", "preconditioned", "m_inverse"]
Solver = Literal["cg", "minres"]
