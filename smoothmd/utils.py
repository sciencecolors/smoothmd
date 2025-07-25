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

from collections import deque

import numpy as np


def extract_descriptors(mol_graph, initial_cartesian):
    """
    Extract internal coordinates descriptors from a molecular graph.

    This function analyzes the molecular graph to determine the optimal set of
    internal coordinates (bond-angle-dihedral descriptors) that can be used to
    reconstruct the molecular structure. It identifies which atoms should be
    kept in cartesian coordinates and which can be described using internal
    coordinates relative to other atoms.

    Args:
        mol_graph: networkx.Graph, molecular graph with atoms as nodes and bonds as edges
        initial_cartesian: array-like, indices of atoms that should remain in cartesian coordinates

    Returns:
        tuple: (descriptors, cartesian_indices) where:
            - descriptors: numpy array of shape (4, M) with indices of atoms A, B, C, D
              for M internal coordinate descriptors (A-B-C-D dihedral relationships, where atom D
              is placed given atoms A, B, C were already placed)
            - cartesian_indices: numpy array of indices for atoms kept in cartesian coordinates
    """
    backbones_graph = mol_graph.subgraph(initial_cartesian)
    resolved = np.zeros(mol_graph.number_of_nodes(), dtype=bool)
    resolved[initial_cartesian] = True
    atoms_a = []
    atoms_b = []
    atoms_c = []
    atoms_d = []
    descriptors = (atoms_a, atoms_b, atoms_c, atoms_d)
    reference_edges = deque(backbones_graph.edges())

    while reference_edges:
        atom2, atom3 = reference_edges.popleft()

        for atom1 in mol_graph[atom2]:
            if atom1 == atom3:
                continue

            for atom4 in mol_graph[atom3]:
                missing = False
                added = False

                if atom4 == atom2:
                    continue

                elif not resolved[atom1] and not resolved[atom4]:
                    missing = True
                    continue

                elif resolved[atom1] and not resolved[atom4]:
                    # descriptors.append((atom1, atom2, atom3, atom4))
                    atoms_a.append(atom1)
                    atoms_b.append(atom2)
                    atoms_c.append(atom3)
                    atoms_d.append(atom4)
                    reference_edges.append((atom3, atom4))
                    resolved[atom4] = True
                    added = True

                elif resolved[atom4] and not resolved[atom1]:
                    # descriptors.append((atom4, atom3, atom2, atom1))
                    atoms_a.append(atom4)
                    atoms_b.append(atom3)
                    atoms_c.append(atom2)
                    atoms_d.append(atom1)
                    reference_edges.append((atom1, atom2))
                    resolved[atom1] = True
                    added = True

                if missing and added:
                    # We may have added a new edge that was needed by the
                    # atom with missing dihedral, so we need to recheck
                    reference_edges.appendleft((atom2, atom3))


    cartesian_indices = np.concatenate((np.where(~resolved)[0], initial_cartesian))
    return np.array(descriptors, dtype=np.int64), cartesian_indices


def compute_angle(out, pos_a, pos_b, pos_c):
    """
    Compute angle ABC (at atom B) from cartesian coordinates.

    Args:
        out: numpy array of shape (N,) to store computed angles
        pos_a: numpy array of shape (N, 3) representing atomic positions of atoms A
        pos_b: numpy array of shape (N, 3) representing atomic positions of atoms B
        pos_c: numpy array of shape (N, 3) representing atomic positions of atoms C

    Returns:
        None: Modifies out array in-place with angles in radians
    """
    vec_ba = pos_a - pos_b
    vec_bc = pos_c - pos_b
    norm_ba = np.linalg.norm(vec_ba, axis=1)
    norm_bc = np.linalg.norm(vec_bc, axis=1)

    # Handle degenerate cases
    non_zero = (norm_ba > 1e-5) & (norm_bc > 1e-5)

    # Use normalized vectors to prevent overflow or underflow
    vec_ba_norm = vec_ba[non_zero] / norm_ba[non_zero, None]
    vec_bc_norm = vec_bc[non_zero] / norm_bc[non_zero, None]
    cos_angle = np.vecdot(vec_ba_norm, vec_bc_norm)

    # Clamp to valid range [-1, 1] to handle numerical errors
    np.clip(cos_angle, -1.0, 1.0, cos_angle)

    # Compute angle
    out[:] = 0.0
    out[non_zero] = np.arccos(cos_angle)


def compute_dihedral(out, pos_a, pos_b, pos_c, pos_d):
    """
    Compute dihedral angle A-B-C-D from cartesian coordinates.

    The dihedral angle is the angle between planes ABC and BCD,
    measured around the B-C bond axis.

    Args:
        out: numpy array of shape (2, N) to store dihedral components (sine, cosine)
        pos_a: numpy array of shape (N, 3) representing atomic positions of atoms A
        pos_b: numpy array of shape (N, 3) representing atomic positions of atoms B
        pos_c: numpy array of shape (N, 3) representing atomic positions of atoms C
        pos_d: numpy array of shape (N, 3) representing atomic positions of atoms D

    Returns:
        None: Modifies out array in-place with dihedral angle components (sine and cosine)
              for numerical stability, range [-π, π]
    """
    # Vectors along the bonds
    vec_ab = pos_b - pos_a # A -> B
    vec_bc = pos_c - pos_b # B -> C
    vec_cd = pos_d - pos_c # C -> D
    normal1 = np.cross(vec_ab, vec_bc) # Normal to plane ABC
    normal2 = np.cross(vec_bc, vec_cd) # Normal to plane BCD

    # Normalize vectors and remove degenerate cases (collinear atoms)
    norm1 = np.linalg.norm(normal1, axis=1)
    norm2 = np.linalg.norm(normal2, axis=1)
    non_zero = (norm1 > 1e-5) & (norm2 > 1e-5)
    normal1 = normal1[non_zero] / norm1[non_zero, None]
    normal2 = normal2[non_zero] / norm2[non_zero, None]

    # Return the dihedral angle as components for numerical stability
    # The sign is determined by the scalar triple product
    vec_bc_u = vec_bc / np.linalg.norm(vec_bc, axis=1)[non_zero, None]

    out[:] = 0.0
    # Sine component
    out[0, non_zero] = np.vecdot(np.cross(normal1, normal2), vec_bc_u)
    # Cosine component
    out[1, non_zero] = np.vecdot(normal1, normal2)


def compute_internal_coordinates(positions, descriptors, cartesian_indices, out_positions, out_lengths, out_angles, out_dihedrals):
    """
    Compute internal coordinates from cartesian positions.

    Args:
        positions: numpy array of shape (N, 3) with atomic positions
        descriptors: numpy array of shape (4, M) with indices of atoms A, B, C, D
        cartesian_indices: numpy array of shape (K,) with indices of cartesian atoms
        out_positions: numpy array of shape (K, 3) to store fixed positions
        out_lengths: numpy array of shape (M,) to store bond lengths
        out_angles: numpy array of shape (M,) to store angles in radians
        out_dihedrals: numpy array of shape (M,) to store dihedral angles in radians
    """
    out_positions[:] = positions[cartesian_indices]

    pos_a = positions[descriptors[0, :]].astype(np.float64)
    pos_b = positions[descriptors[1, :]].astype(np.float64)
    pos_c = positions[descriptors[2, :]].astype(np.float64)
    pos_d = positions[descriptors[3, :]].astype(np.float64)

    out_lengths[:] = np.linalg.norm(pos_c - pos_d, axis=1)
    compute_angle(out_angles, pos_b, pos_c, pos_d)
    compute_dihedral(out_dihedrals, pos_a, pos_b, pos_c, pos_d)
