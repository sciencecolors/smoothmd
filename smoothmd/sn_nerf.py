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

import numpy as np


def SN_NeRF(out_positions, descriptors, lengths, angles, dihedrals):
    """
    Compute new positions using the SN-NeRF method.

    Args:
        out_positions: numpy array of shape (N, 3) with fixed positions already computed
        descriptors: numpy array of shape (4, M) with indices of atoms A, B, C, D
        lengths: numpy array of shape (M,) with bond lengths
        angles: numpy array of shape (M,) with angles in radians
        dihedrals: numpy array of shape (M,) with dihedral angles in radians

    Returns:
        None: Modifies out_positions in-place to include computed atomic positions
    """
    for i in range(descriptors.shape[1]):
        a = descriptors[0, i]
        b = descriptors[1, i]
        c = descriptors[2, i]
        d = descriptors[3, i]
        out_positions[d] = place_atom(out_positions[a], out_positions[b], out_positions[c], lengths[i], angles[i], dihedrals[i])


def place_atom(pos_a, pos_b, pos_c, length, theta, phi):
    """
    Place a new atom D given three reference atoms A, B, C and respective internal coordinates.

    This function implements the core coordinate reconstruction using the SN-NeRF
    (Self-Normalizing Natural Extension Reference Frame) method to place atom D relative to
    atoms A, B, and C using bond length, angle, and dihedral angle.

    Reference:
        - Parsons J, Holmes JB, Rojas JM, Tsai J, Strauss CE. Practical conversion from torsion
          space to Cartesian space for in silico protein synthesis. J Comput Chem. 2005 Jul
          30;26(10):1063-8. doi: 10.1002/jcc.20237. PMID: 15898109.

    Args:
        pos_a: numpy array of shape (3,) with position of atom A
        pos_b: numpy array of shape (3,) with position of atom B
        pos_c: numpy array of shape (3,) with position of atom C
        length: float, bond length C-D
        theta: float, bond angle B-C-D in radians
        phi: float, dihedral angle A-B-C-D in radians

    Returns:
        numpy array of shape (3,) with the computed position of atom D
    """
    # Base vectors AB and BC
    ab = (pos_b - pos_a)
    bc_unit = (pos_c - pos_b) / np.linalg.norm(pos_c - pos_b)

    # Normals n = AB x BC, p = n x BC
    n_unit = np.cross(ab, bc_unit)
    n_unit = n_unit / np.linalg.norm(n_unit)
    p_unit = np.cross(n_unit, bc_unit)

    # Rotation matrix [BC; p;  n]
    rotation = np.empty((3, 3), dtype=np.float64)
    rotation[:, 0] = bc_unit
    rotation[:, 1] = p_unit
    rotation[:, 2] = n_unit

    # Local frame
    sin_theta = np.sin(theta)
    d2 = np.empty(3, dtype=np.float64)
    d2[0] = -length * np.cos(theta)
    d2[1] = length * sin_theta * np.cos(phi)
    d2[2] = length * sin_theta * np.sin(phi)

    # Rotate and translate to get global coordinates
    return pos_c + np.dot(rotation, d2)
