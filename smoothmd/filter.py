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
import networkx as nx
import MDAnalysis as mda

from .sn_nerf import SN_NeRF
from .utils import extract_descriptors, compute_internal_coordinates


def FILTER_GMX(x):
    """
    Low pass filter function used by Gromacs gmx filter.
    It is defined as `cos(pi t/A) + 1` where `A` is the period of the filtered
    movements and `t` is the filter window step.
    """
    return 1 + np.cos(x * np.pi)


def filter_smooth_trajectory(out_trajectory, universe, cartesian_indices=[], period=0, step_frames=1, filter_func=FILTER_GMX, verbose=True):
    """
    Apply smoothing filter to a molecular dynamics trajectory using internal coordinates.

    This function uses temporal filtering to smooth the trajectory of a molecular dynamics
    simulation. The filtering is performed in both cartesian and internal coordinates space
    (bond lengths, angles, and dihedral angles). Filtering window is defined as
    2 * period + 1 frames, and thus the output trajectory will have 2 * period
    fewer frames.

    By operating in internal coordinates, the method avoids issues with smoothing rotational
    degrees of freedom and ensures that the resulting trajectory remains physically meaningful.

    For improved numerical stability, it is advised to smooth backbone atoms in cartesian
    coordinates (see `cartesian_indices` argument). The same applies to other organic molecules.

    The function detects all atoms that can be described using internal coordinates relative to
    the backbone atoms. Any atom that can't be described in internal coordinates will be treated
    in cartesian space. At least three atoms must be specified in `cartesian_indices` to define a
    valid reference frame.

    Args:
        out_trajectory: str, path to output trajectory file. Supported formats are those supported
                        by MDAnalysis.
        universe: MDAnalysis.Universe, input trajectory universe
        cartesian_indices: list of int, indices of atoms to keep in cartesian coordinates
                           (default: empty list)
        period: int, period of movements to be smoothed out (default: 0, no smoothing)
        step_frames: int, number of frames to advance between writes in the output. Useful for
                     reducing trajectory size (default: 1, keep all frames).
        filter_func: callable, function that defines the weights of the filter window. It must
                     accept a 1D numpy array as input with values in the range [-0.5, 0.5] and
                     return the array of corresponding weights. Weights are normalized to sum 1.
                     Default is a cosine filter used by Gromacs gmx filter (FILTER_GMX).
        verbose: bool, whether to print progress information (default: True)

    Returns:
        None: Writes smoothed trajectory to out_trajectory file
    """
    atoms = universe.atoms
    num_atoms = atoms.n_atoms
    num_frames = universe.trajectory.n_frames

    # Build graph
    mol_graph = nx.Graph()
    mol_graph.add_nodes_from(atoms.indices)
    mol_graph.add_edges_from(universe.bonds.to_indices())

    # Extract descriptors
    descriptors, cartesian_indices = extract_descriptors(mol_graph, cartesian_indices)
    num_cartesian = cartesian_indices.shape[0]
    num_internal = descriptors.shape[1]

    # Filter data
    double_period = period * 2
    filter_size = double_period + 1
    weights = filter_func(np.linspace(-period, period, filter_size) / filter_size)
    weights = weights / weights.sum()

    # Arrays for storing moving windows
    cartesian_window = np.empty((num_cartesian, 3, filter_size), dtype=np.float64)
    lengths_window = np.empty((num_internal, filter_size), dtype=np.float64)
    angles_window = np.empty_like(lengths_window)
    # For dihedrals, filter sine and cosine components separately to avoid wrapping issues
    dihedrals_window = np.empty((2, num_internal, filter_size), dtype=np.float64)  # (sine, cosine)

    # Temporary arrays to avoid recreating them in the loop
    filtered_positions = np.empty((num_atoms, 3), dtype=np.float64)
    filtered_lengths = np.empty(num_internal, dtype=np.float64)
    filtered_angles = np.empty_like(filtered_lengths, dtype=np.float64)
    convoluted_dihedrals = np.empty((2, num_internal), dtype=np.float64)  # (sine, cosine)
    filtered_dihedrals = np.empty_like(filtered_lengths, dtype=np.float64)

    with mda.Writer(out_trajectory, num_atoms) as writer:
        out_universe = mda.Universe.empty(num_atoms, trajectory=True)

        # Fill initial window
        for frame in range(filter_size - 1):
            if verbose:
                print(f"\rSmooth: Processing frame {frame + 1} of {num_frames}", end="")

            positions = universe.trajectory[frame].positions

            compute_internal_coordinates(
                positions,
                descriptors,
                cartesian_indices,
                cartesian_window[:, :, frame],
                lengths_window[:, frame],
                angles_window[:, frame],
                dihedrals_window[:, :, frame]
            )

        # Process trajectory
        weights_index = np.arange(filter_size)
        for out_frame, frame in enumerate(range(filter_size - 1, num_frames)):
            if verbose:
                print(f"\rSmooth: Processing frame {frame + 1} of {num_frames}", end="")

            positions = universe.trajectory[frame].positions
            index = frame % filter_size

            # Consider all frames into the windows
            compute_internal_coordinates(
                positions,
                descriptors,
                cartesian_indices,
                cartesian_window[:, :, index],
                lengths_window[:, index],
                angles_window[:, index],
                dihedrals_window[:, :, index]
            )

            # Process only frames that will be output
            if out_frame % step_frames == 0:
                # Filtering
                reordered_weights = weights[weights_index]
                filtered_positions[cartesian_indices] = np.vecdot(cartesian_window, reordered_weights)
                np.vecdot(lengths_window, reordered_weights, out=filtered_lengths)
                np.vecdot(angles_window, reordered_weights, out=filtered_angles)
                np.vecdot(dihedrals_window, reordered_weights, out=convoluted_dihedrals)
                np.arctan2(convoluted_dihedrals[0], convoluted_dihedrals[1], out=filtered_dihedrals)

                # Convert back to cartesian coordinates
                SN_NeRF(filtered_positions, descriptors, filtered_lengths, filtered_angles, filtered_dihedrals)
                out_universe.trajectory.ts.positions = filtered_positions
                writer.write(out_universe.atoms)

            # Shift weights indexing
            weights_index += double_period
            weights_index %= filter_size

        if verbose:
            print()
