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
ALL_COLORS: dict[str, str] = mcolors.TABLEAU_COLORS
ALL_COLORS.update(mcolors.XKCD_COLORS)
for d in [mcolors.BASE_COLORS, mcolors.CSS4_COLORS]:
    for k, v in d.items():
        ALL_COLORS[k] = mcolors.to_hex(v)
ALL_COLORS_REVERSE = {v: k for k, v in ALL_COLORS.items()}


def get_matplotlib_color(identifier: str) -> tuple[str, str]:
    """Return a valid Matplotlib color by an identifier recognized by
    :func:`~matplotlib.colors.is_color_like`.

    Parameters
    ----------
    identifier
        String identifying a color.

    Returns
    -------
    name
        Matplotlib color name.
    hex_value
        Hex color string.
    """
    hex_value = mcolors.to_hex(identifier)  # Raises if invalid
    name = ALL_COLORS_REVERSE[hex_value]
    return name, hex_value
