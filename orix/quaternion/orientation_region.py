# -*- coding: utf-8 -*-
# Copyright 2018-2023 the orix developers
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix.  If not, see <http://www.gnu.org/licenses/>.

# The below EMsoft copyright notice is included because the following
# functionality in this file is derived from EMsoft's source code:
#  - Determination of whether a Rodrigues vector lies inside the
#    fundamental zone

# #####################################################################
# Copyright (c) 2013-2023, Marc De Graef Research Group/Carnegie Mellon
# University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the
#     distribution.
#   - Neither the names of Marc De Graef, Carnegie Mellon University nor
#     the names of its contributors may be used to endorse or promote
#     products derived from this software without specific prior written
#     permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ###################################################################

"""An orientation region is some subset of the complete space of orientations.

The complete orientation space represents every possible orientation of an
object. The whole space is not always needed, for example if the orientation
of an object is constrained or (most commonly) if the object is symmetrical. In
this case, the space can be segmented using sets of Rotations representing
boundaries in the space. This is clearest in the Rodrigues parametrisation,
where the boundaries are planes, such as the example here: the asymmetric
domain of an adjusted 432 symmetry.

.. image:: /_static/img/orientation-region-Oq.png
   :width: 300px
   :alt: Boundaries of an orientation region in Rodrigues space.
   :align: center

Rotations or orientations can be inside or outside of an orientation region.
"""

from __future__ import annotations

import itertools
from typing import Tuple

import numba as nb
import numpy as np

from orix.quaternion import Quaternion
from orix.quaternion.rotation import Rotation
from orix.quaternion.symmetry import C1, Symmetry, get_distinguished_points
from orix.vector import Rodrigues, Vector3d

# Constants
_EPS = 1e-9
_SQRT3 = np.sqrt(3)
_ONE_OVER_SQRT2 = 1 / np.sqrt(2)
_SQRT3_OVER2 = np.sqrt(3) / 2
_TOL = 1e5
_TAN_PI_OVER_2N = np.zeros(5, dtype=np.float64)
_TAN_PI_OVER_2N[1:5] = np.tan(np.pi / (2 * np.array([2, 3, 4, 6])))
_TAN_PI_OVER_2N = np.insert(_TAN_PI_OVER_2N, 4, 0)


def _get_large_cell_normals(s1, s2):
    dp = get_distinguished_points(s1, s2)
    normals = Rodrigues.zero(dp.shape + (2,))
    planes1 = dp.axis * np.tan(dp.angle / 4)
    planes2 = -dp.axis * np.tan(dp.angle / 4) ** -1
    planes2.data[np.isnan(planes2.data)] = 0
    normals[:, 0] = planes1
    normals[:, 1] = planes2
    normals: Rotation = (
        Rotation.from_neo_euler(normals).flatten().unique(antipodal=False)
    )
    if not normals.size:
        return normals
    _, inv = normals.axis.unique(return_inverse=True)
    axes_unique = []
    angles_unique = []
    for i in np.unique(inv):
        n = normals[inv == i]
        axes_unique.append(n.axis.data[0])
        angles_unique.append(np.max(n.angle))
    normals = Rotation.from_axes_angles(np.array(axes_unique), angles_unique)
    return normals


def get_proper_groups(Gl: Symmetry, Gr: Symmetry) -> Tuple[Symmetry, Symmetry]:
    """Return the appropriate groups for the asymmetric domain
    calculation.

    Parameters
    ----------
    Gl
        First point group.
    Gr
        Second point group.

    Returns
    -------
    Gl
        First proper subgroup(s) or proper inversion subgroup(s), as
        appropriate.
    Gr
        Second proper subgroup(s) or proper inversion subgroup(s), as
        appropriate.

    Raises
    ------
    NotImplementedError
        If both groups are improper and neither contain an inversion,
        special consideration is needed which is not yet implemented in
        orix.
    """
    if Gl.is_proper and Gr.is_proper:
        return Gl, Gr
    elif Gl.is_proper and not Gr.is_proper:
        return Gl, Gr.proper_subgroup
    elif not Gl.is_proper and Gr.is_proper:
        return Gl.proper_subgroup, Gr
    else:
        if Gl.contains_inversion and Gr.contains_inversion:
            return Gl.proper_subgroup, Gr.proper_subgroup
        elif Gl.contains_inversion and not Gr.contains_inversion:
            return Gl.proper_subgroup, Gr.laue_proper_subgroup
        elif not Gl.contains_inversion and Gr.contains_inversion:
            return Gl.laue_proper_subgroup, Gr.proper_subgroup
        else:
            raise NotImplementedError(
                "Both groups are improper, and do not contain inversion"
            )


class OrientationRegion(Rotation):
    """A set of :class:`~orix.quaternion.Rotation` which are the normals
    of an orientation region.
    """

    @classmethod
    def from_symmetry(cls, s1: Symmetry, s2: Symmetry = C1) -> OrientationRegion:
        """The set of unique (mis)orientations of a symmetrical object.

        Parameters
        ----------
        s1
            First symmetry.
        s2
            Second symmetry.
        """
        s1, s2 = get_proper_groups(s1, s2)
        large_cell_normals = _get_large_cell_normals(s1, s2)
        disjoint = s1 & s2
        fz = disjoint.fundamental_zone()
        fz_normals = Rotation.from_axes_angles(fz, np.pi)
        normals = Rotation(np.concatenate([large_cell_normals.data, fz_normals.data]))
        orientation_region = cls(normals)
        vertices = orientation_region.vertices()
        if vertices.size:
            orientation_region = orientation_region[
                np.any(np.isclose(orientation_region.dot_outer(vertices), 0), axis=1)
            ]
        return orientation_region

    def vertices(self) -> Rotation:
        """Return the vertices of the asymmetric domain.

        Returns
        -------
        rot
            Domain vertices.
        """
        normal_combinations = list(itertools.combinations(self, 3))
        if len(normal_combinations) < 1:
            return Rotation.empty()
        c1, c2, c3 = zip(*normal_combinations)
        c1, c2, c3 = (
            Rotation.stack(c1).flatten(),
            Rotation.stack(c2).flatten(),
            Rotation.stack(c3).flatten(),
        )
        rot = Rotation.triple_cross(c1, c2, c3)
        rot = rot[~np.any(np.isnan(rot.data), axis=-1)]
        rot = rot[rot < self].unique()
        surface = np.any(np.isclose(rot.dot_outer(self), 0), axis=1)
        return rot[surface]

    def faces(self) -> list:
        normals = Rotation(self)
        vertices = self.vertices()
        faces = []
        for n in normals:
            faces.append(vertices[np.isclose(vertices.dot(n), 0)])
        faces = [f for f in faces if f.size > 2]
        return faces

    def __gt__(self, other: OrientationRegion) -> np.ndarray:
        """Overridden greater than method. Applying this to an
        Orientation will return only those orientations that lie within
        the OrientationRegion.
        """
        c = Quaternion(self).dot_outer(Quaternion(other))
        inside = np.logical_or(
            np.all(np.greater_equal(c, -_EPS), axis=0),
            np.all(np.less_equal(c, +_EPS), axis=0),
        )
        return inside

    def get_plot_data(self) -> Rotation:
        """Suitable Rotations for the construction of a wireframe."""

        # Get a grid of vector directions
        azimuth = np.linspace(0, 2 * np.pi - _EPS, 361)
        polar = np.linspace(0, np.pi - _EPS, 181)
        aa, pp = np.meshgrid(azimuth, polar)
        g = Vector3d.from_polar(aa, pp)

        # Get the cell vector normal norms
        n = Rodrigues.from_rotation(self).norm[:, np.newaxis, np.newaxis]
        if n.size == 0:
            return Rotation.from_axes_angles(g, np.pi)

        d = (-self.axis).dot_outer(g.unit)
        x = n * d
        with np.errstate(divide="ignore"):
            omega = 2 * np.arctan(np.where(x != 0, x**-1, np.pi))

        # Keep the smallest allowed angle
        omega[omega < 0] = np.pi
        omega = np.min(omega, axis=0)
        r = Rotation.from_axes_angles(g.unit, omega)

        return r


# The following are logical functions to determine whether a Rodrigues
# vector lies inside a fundamental zone of one of the eleven
# crystallographic proper point groups. The implementation is based on
# the one in EMsoft's S0(3) module, which itself is based on the paper
# by Morawiec and Field (1996), doi: 10.1080/01418619608243708.


@nb.njit("bool_(float64[:], int64, int64)", cache=True, fastmath=True)
def _is_inside_cyclic_fz(ro: np.ndarray, fz_type: int, fz_order: int) -> bool:
    """Return whether a Rodrigues vector is within the specified
    cyclic Rodrigues fundamental zone :cite:`morawiec1996rodrigues`.

    The fundamental zone of Cn (n = 2, 3, 4, 6) is the full space
    bounded by two planes perpendicular to the n-fold axis, each at the
    distance tan(pi / 2n) from the origin.

    Parameters
    ----------
    ro
        Rodrigues vector components (x, y, z, angle) as 64-bit floats.
    fz_order
        Order of the fundamental zone, Cn.

    Returns
    -------
    is_inside
        Whether the vector is inside the specified cyclic Rodrigues
        fundamental zone.
    """
    if ro[3] != np.inf:
        fz_extent = _TAN_PI_OVER_2N[fz_order - 1]
        if fz_type == 1 and fz_order == 2:
            # Is Ry within tan(pi/2n)?
            is_inside = abs(ro[1] * ro[3]) <= fz_extent
        else:
            # Is Rz within tan(pi/2n)?
            is_inside = abs(ro[2] * ro[3]) <= fz_extent
    else:
        if fz_type == 1 and fz_order == 2:
            is_inside = abs(ro[1]) <= _EPS
        else:
            is_inside = abs(ro[2]) <= _EPS

    return is_inside


@nb.njit("bool_(float64[:], int64)", cache=True, fastmath=True, nogil=True)
def _is_inside_dihedral_fz(ro: np.ndarray, fz_order: int) -> bool:
    """Return whether a Rodrigues vector is within the specified
    dihedral Rodrigues fundamental zone :cite:`morawiec1996rodrigues`.

    The fundamental zone of Dn (n = 2, 3, 4, 6) is a prism with 2n-sided
    polygons (at distance tan(pi / 2n) from the origin) as prism bases,
    and 2n square prism faces at a distance tan(pi / 4) = 1 from the
    origin. The bases are perpendicular to the n-fold axis and the faces
    are perpendicular to the 2-fold axes.

    Parameters
    ----------
    ro
        Rodrigues vector components (x, y, z, angle) as 64-bit floats.
    fz_order
        Order of the fundamental zone, Dn.

    Returns
    -------
    is_inside
        Whether the vector is inside the specified dihedral Rodrigues
        fundamental zone.
    """
    if ro[3] > np.sqrt(3):
        is_inside = False
    else:
        rx, ry, rz = ro[:3] * ro[3]

        # Is Rz within prism bases, tan(pi / 2n)?
        cond1 = abs(rz) <= _TAN_PI_OVER_2N[fz_order - 1]
        is_inside = False

        # fmt: off
        if cond1:
            # Are Rx and Ry within the square prism faces?
            if fz_order == 2:
                cond2 = (
                        abs(rx) <= 1.
                    and abs(rz) <= 1.
                )
            elif fz_order == 3:
                cond2 = (
                        abs(rx) <= 1.
                    and abs(_SQRT3_OVER2 * ry + 0.5 * rx) <= 1.
                    and abs(_SQRT3_OVER2 * ry - 0.5 * rx) <= 1.
                )
            elif fz_order == 4:
                cond2 = (
                        abs(rx) <= 1.
                    and abs(ry) <= 1.
                    and _ONE_OVER_SQRT2 * abs(ry + rx) <= 1.
                    and _ONE_OVER_SQRT2 * abs(ry - rx) <= 1.
                )
            else:  # fz_order == 6
                cond2 = (
                        abs(rx) <= 1.
                    and abs(ry) <= 1.
                    and abs(0.5 * ry + _SQRT3_OVER2 * rx) <= 1.
                    and abs(_SQRT3_OVER2 * ry + 0.5 * rx) <= 1.
                    and abs(_SQRT3_OVER2 * ry - 0.5 * rx) <= 1.
                    and abs(0.5 * ry - _SQRT3_OVER2 * rx) <= 1.
                )
            is_inside = cond2
        # fmt: on

    return is_inside


@nb.njit("bool_(float64[:], bool_)", cache=True, fastmath=True, nogil=True)
def _is_inside_cubic_fz(ro: np.ndarray, octahedral: bool) -> bool:
    """Return whether a Rodrigues vector is within the tetrahedral or
    octahedral Rodrigues fundamental zone :cite:`morawiec1996rodrigues`.

    The tetrahedral fundamental zone is a regular octahedron with faces
    at distances tan(pi / 6) from the origin which are perpendicular to
    the 3-fold axes. The octahedral fundamental zone is a truncated cube
    with six octagonal faces at distances tan(pi / 4) = 1 from the
    origin and eight triangular faces at distances tan(pi / 6) from the
    origin.

    Parameters
    ----------
    ro
        Rodrigues vector components (x, y, z, angle) as 64-bit floats.
    octahedral
        Whether the vector should be checked for the octahedral symmetry
        as well as the tetrahedral one. Default is False.

    Returns
    -------
    is_inside
        Whether the vector is inside the tetrahedral or octahedral
        Rodrigues fundamental zone.
    """
    fz_extent = _TAN_PI_OVER_2N[3]
    ro_axis = np.abs(ro[:3] * ro[3])

    if octahedral:
        cond1 = np.max(ro_axis) <= fz_extent
    else:
        cond1 = True

    cond2 = np.sum(ro_axis) <= 1.0

    return cond1 and cond2


@nb.njit("bool_(float64[:], int64, int64)", cache=True, fastmath=True, nogil=True)
def _is_inside_fz(ro: np.ndarray, fz_type: int, fz_order: int) -> bool:
    if fz_type == 0:
        is_inside = True
    elif fz_type == 1:
        is_inside = _is_inside_cyclic_fz(ro, fz_type, fz_order)
    elif fz_type == 2:
        is_inside = _is_inside_dihedral_fz(ro, fz_order)
    elif fz_type == 3:  # Tetrahedral symmetry
        is_inside = _is_inside_cubic_fz(ro, False)
    elif fz_type == 4:
        is_inside = _is_inside_cubic_fz(ro, True)
    else:
        is_inside = False

    return is_inside
