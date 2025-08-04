"""
Unit tests for smoothmd.sn_nerf module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from smoothmd.sn_nerf import SN_NeRF, place_atom
from smoothmd.utils import compute_angle, compute_dihedral


def check_atom(pos_a, pos_b, pos_c, pos_d, expected_length, expected_angle, expected_dihedral, rtol=1e-7, atol=1e-8):
    """
    Helper function to check if placed atom D has correct internal coordinates.

    Args:
        pos_a: numpy array of shape (3,) with position of atom A
        pos_b: numpy array of shape (3,) with position of atom B
        pos_c: numpy array of shape (3,) with position of atom C
        pos_d: numpy array of shape (3,) with position of atom D
        expected_length: float, expected bond length C-D
        expected_angle: float, expected bond angle B-C-D in radians
        expected_dihedral: float, expected dihedral angle A-B-C-D in radians
        rtol: float, relative tolerance for assertions
        atol: float, absolute tolerance for assertions
    """
    # Check bond length C-D
    length = np.linalg.norm(pos_d - pos_c)
    assert_allclose(length, expected_length, rtol=rtol, atol=atol)

    # Check angle B-C-D
    angle = np.zeros(1)
    compute_angle(angle, pos_b[None, :], pos_c[None, :], pos_d[None, :])
    assert_allclose(angle[0], expected_angle, rtol=rtol, atol=atol)

    # Check dihedral angle A-B-C-D
    dihedral_out = np.zeros((2, 1))
    compute_dihedral(dihedral_out, pos_a[None, :], pos_b[None, :], pos_c[None, :], pos_d[None, :])
    dihedral = np.arctan2(dihedral_out[0, 0], dihedral_out[1, 0])
    assert_allclose(dihedral, expected_dihedral, rtol=rtol, atol=atol)


class TestPlaceAtom:
    """Test the place_atom function."""

    def test_place_along_x_axis(self):
        """Test placing an atom with non-collinear reference atoms."""
        # Use non-collinear reference atoms to avoid numerical issues
        pos_a = np.array([0.0, 0.0, 0.0])
        pos_b = np.array([1.0, 0.0, 0.0])
        pos_c = np.array([1.0, 1.0, 0.0])  # Forms a right angle at B

        # Place atom at distance 1.0, angle 90°, dihedral 0°
        length = 1.0
        theta = np.pi / 2  # 90 degrees
        phi = 0.0          # 0 degrees dihedral

        pos_d = place_atom(pos_a, pos_b, pos_c, length, theta, phi)

        check_atom(pos_a, pos_b, pos_c, pos_d, length, theta, phi)

    def test_place_perpendicular(self):
        """Test placing an atom perpendicular to a bond."""
        # Use non-collinear reference atoms
        pos_a = np.array([0.0, 0.0, 0.0])
        pos_b = np.array([1.0, 0.0, 0.0])
        pos_c = np.array([1.0, 1.0, 0.0])  # Forms a right angle at B

        # Place atom at distance 1.0, angle 90°, dihedral 90°
        length = 1.0
        theta = np.pi / 2  # 90 degrees
        phi = np.pi / 2    # 90 degrees dihedral

        pos_d = place_atom(pos_a, pos_b, pos_c, length, theta, phi)

        check_atom(pos_a, pos_b, pos_c, pos_d, length, theta, phi)

    def test_tetrahedral_geometry(self):
        """Test placing atoms in tetrahedral geometry."""
        # Set up a carbon with three hydrogens
        pos_c = np.array([0.0, 0.0, 0.0])
        pos_h1 = np.array([1.0, 0.0, 0.0])
        pos_h2 = np.array([0.0, 1.0, 0.0])

        # Place fourth hydrogen at tetrahedral angle
        length = 1.0
        theta = np.arccos(-1.0/3.0)  # Tetrahedral angle ≈ 109.47°
        phi = 0.0

        pos_h3 = place_atom(pos_c, pos_h1, pos_h2, length, theta, phi)

        check_atom(pos_c, pos_h1, pos_h2, pos_h3, length, theta, phi)

    def test_different_bond_lengths(self):
        """Test with different bond lengths."""
        # Use non-collinear reference atoms
        pos_a = np.array([0.0, 0.0, 0.0])
        pos_b = np.array([1.0, 0.0, 0.0])
        pos_c = np.array([1.0, 1.0, 0.0])

        theta = np.pi / 2
        phi = 0.0

        for length in [0.5, 1.0, 1.5, 2.0]:
            pos_d = place_atom(pos_a, pos_b, pos_c, length, theta, phi)
            check_atom(pos_a, pos_b, pos_c, pos_d, length, theta, phi)

    def test_different_angles(self):
        """Test with different bond angles."""
        # Use non-collinear reference atoms
        pos_a = np.array([0.0, 0.0, 0.0])
        pos_b = np.array([1.0, 0.0, 0.0])
        pos_c = np.array([1.0, 1.0, 0.0])
        length = 1.0
        phi = 0.0

        for theta in [np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3]:
            pos_d = place_atom(pos_a, pos_b, pos_c, length, theta, phi)
            check_atom(pos_a, pos_b, pos_c, pos_d, length, theta, phi)

    def test_dihedral_rotation(self):
        """Test that dihedral angle rotates atom correctly."""
        # Use non-collinear reference atoms
        pos_a = np.array([0.0, 0.0, 0.0])
        pos_b = np.array([1.0, 0.0, 0.0])
        pos_c = np.array([1.0, 1.0, 0.0])
        length = 1.0
        theta = np.pi / 2

        # Test different dihedral angles with expected values in [-π, π] range
        test_cases = [
            (0.0, 0.0),              # 0° -> 0°
            (np.pi / 2, np.pi / 2),  # 90° -> 90°
            (np.pi, np.pi),          # 180° -> 180° (or -180°, both equivalent)
            (3 * np.pi / 2, -np.pi / 2)  # 270° -> -90° (equivalent angles)
        ]

        for input_phi, expected_phi in test_cases:
            pos_d = place_atom(pos_a, pos_b, pos_c, length, theta, input_phi)
            check_atom(pos_a, pos_b, pos_c, pos_d, length, theta, expected_phi)

    def test_small_bond_length_cases(self):
        """Test small bond length edge case."""
        # Use non-collinear reference atoms
        pos_a = np.array([0.0, 0.0, 0.0])
        pos_b = np.array([1.0, 0.0, 0.0])
        pos_c = np.array([1.0, 1.0, 0.0])
        theta = np.pi / 2
        phi = 0.0

        length = 1e-4
        pos_d = place_atom(pos_a, pos_b, pos_c, length, theta, phi)
        check_atom(pos_a, pos_b, pos_c, pos_d, length, theta, phi)


class TestSNNeRF:
    """Test the SN_NeRF function."""

    def test_single_atom_placement(self):
        """Test placing a single atom using SN_NeRF."""
        # Initial positions for first three atoms (non-collinear)
        out_positions = np.zeros((4, 3))
        out_positions[0] = [0.0, 0.0, 0.0]
        out_positions[1] = [1.0, 0.0, 0.0]
        out_positions[2] = [1.0, 1.0, 0.0]  # Forms a right angle

        # Descriptor to place atom 3
        descriptors = np.array([[0], [1], [2], [3]], dtype=np.int64)
        lengths = np.array([1.0])
        angles = np.array([np.pi / 2])
        dihedrals = np.array([0.0])

        SN_NeRF(out_positions, descriptors, lengths, angles, dihedrals)

        # Verify that atom 3 was placed correctly
        check_atom(out_positions[0], out_positions[1], out_positions[2], out_positions[3],
                  lengths[0], angles[0], dihedrals[0])

    def test_multiple_atom_placement(self):
        """Test placing multiple atoms sequentially."""
        # Set up for 5 atoms, first 3 fixed (non-collinear)
        out_positions = np.zeros((5, 3))
        out_positions[0] = [0.0, 0.0, 0.0]
        out_positions[1] = [1.0, 0.0, 0.0]
        out_positions[2] = [1.0, 1.0, 0.0]

        # Descriptors to place atoms 3, 4
        descriptors = np.array([
            [0, 1],  # A atoms
            [1, 2],  # B atoms
            [2, 3],  # C atoms
            [3, 4],  # D atoms
        ], dtype=np.int64)

        lengths = np.array([1.0, 1.0])
        angles = np.array([np.pi / 2, np.pi / 2])
        dihedrals = np.array([0.0, np.pi / 2])

        SN_NeRF(out_positions, descriptors, lengths, angles, dihedrals)

        # Verify that all atoms were placed correctly
        for i in range(descriptors.shape[1]):
            a_idx = descriptors[0, i]
            b_idx = descriptors[1, i]
            c_idx = descriptors[2, i]
            d_idx = descriptors[3, i]
            check_atom(out_positions[a_idx], out_positions[b_idx], out_positions[c_idx],
                      out_positions[d_idx], lengths[i], angles[i], dihedrals[i])

    def test_empty_descriptors(self):
        """Test with empty descriptors array."""
        out_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        descriptors = np.zeros((4, 0), dtype=np.int64)
        lengths = np.array([])
        angles = np.array([])
        dihedrals = np.array([])

        # Should not raise any errors
        SN_NeRF(out_positions, descriptors, lengths, angles, dihedrals)

    def test_collinear_atoms_error(self):
        """Test that collinear atoms raise appropriate error."""
        # Collinear atoms should raise ValueError
        pos_a = np.array([0.0, 0.0, 0.0])
        pos_b = np.array([1.0, 0.0, 0.0])
        pos_c = np.array([2.0, 0.0, 0.0])  # Collinear with A and B

        with pytest.raises(ValueError, match="collinear"):
            place_atom(pos_a, pos_b, pos_c, 1.0, np.pi / 2, 0.0)

    def test_coincident_atoms_error(self):
        """Test that coincident atoms raise appropriate error."""
        # Coincident B and C atoms should raise ValueError
        pos_a = np.array([0.0, 0.0, 0.0])
        pos_b = np.array([1.0, 0.0, 0.0])
        pos_c = np.array([1.0, 0.0, 0.0])  # Same as B

        with pytest.raises(ValueError, match="too close together"):
            place_atom(pos_a, pos_b, pos_c, 1.0, np.pi / 2, 0.0)


if __name__ == "__main__":
    pytest.main([__file__])
