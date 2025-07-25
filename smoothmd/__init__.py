# Copyright (c) Caio S. Souza and Science Colors
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.

"""
SmoothMD: tools for smoothing molecular dynamics trajectories.

This library provides tools for smoothing molecular dynamics trajectories such as
applying filters in internal coordinates (bonds, angles, dihedrals) space.
"""

from .filter import filter_smooth_trajectory, FILTER_GMX

__version__ = "0.1.0"
