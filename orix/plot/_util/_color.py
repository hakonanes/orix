#
# Copyright 2018-2025 the orix developers
#
# This file is part of orix.
#
# orix is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# orix is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix. If not, see <http://www.gnu.org/licenses/>.
#

"""Private utilities for handling color in visualization."""

import matplotlib.colors as mcolors

# All named Matplotlib colors (tableau and xkcd already lower case hex)
ALL_COLORS = mcolors.TABLEAU_COLORS
for k, v in {**mcolors.BASE_COLORS, **mcolors.CSS4_COLORS}.items():
    ALL_COLORS[k] = mcolors.to_hex(v)
ALL_COLORS.update(mcolors.XKCD_COLORS)
