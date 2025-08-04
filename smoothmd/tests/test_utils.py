"""
Unit tests for smoothmd.utils module.
"""

import numpy as np
import networkx as nx
import pytest
from numpy.testing import assert_allclose

from smoothmd.utils import (
    extract_descriptors,
    compute_angle,
    compute_dihedral,
    compute_internal_coordinates
)


class TestExtractDescriptors:
    """Test the extract_descriptors function."""

    def test_simple_chain(self):
        """Test with a simple linear chain of atoms."""
        # Create a simple chain: 0-1-2-3
        mol_graph = nx.Graph()
        mol_graph.add_edges_from([(0, 1), (1, 2), (2, 3)])

        # Use first three atoms as cartesian reference (minimum needed)
        initial_cartesian = [0, 1, 2]

        descriptors, cartesian_indices = extract_descriptors(mol_graph, initial_cartesian)

        # Should have descriptors in (A, B, C, D) format
        assert descriptors.shape[0] == 4
        # Descriptors for remaining atoms
        assert descriptors.shape[1] == 1

        # Should include initial cartesian atoms
        assert 0 in cartesian_indices
        assert 1 in cartesian_indices
        assert 2 in cartesian_indices

        # Should not include atom 3 in cartesian indices
        assert 3 not in cartesian_indices

        # Descriptors for atom 3 should be (0, 1, 2, 3)
        assert np.array_equal(descriptors[:, 0], [0, 1, 2, 3])

    def test_branched_molecule(self):
        """Test with a branched molecule structure that cannot be described with internal coordinates."""
        # Create branched structure: 0-1-2-3 with 4 bonded to 2
        mol_graph = nx.Graph()
        mol_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 4)])

        # Use first three atoms as cartesian reference (minimum needed)
        initial_cartesian = [0, 1, 2]

        descriptors, cartesian_indices = extract_descriptors(mol_graph, initial_cartesian)

        # Should have descriptors in (A, B, C, D) format
        assert descriptors.shape[0] == 4
        # Two descsriptors for atom 3 and 4
        assert descriptors.shape[1] == 2

        # Should include initial cartesian atoms
        assert 0 in cartesian_indices
        assert 1 in cartesian_indices
        assert 2 in cartesian_indices

        # Should not include atoms 3 and 4 in cartesian indices
        assert 3 not in cartesian_indices
        assert 4 not in cartesian_indices

        # Descriptors for atom 3 should be (0, 1, 2, 3) and for atom 4 (0, 1, 2, 4)
        assert np.array_equal(descriptors[:, 0], [0, 1, 2, 3])
        assert np.array_equal(descriptors[:, 1], [0, 1, 2, 4])

    def test_no_descriptors_branched_molecule(self):
        """Test with a branched molecule structure that cannot be described with internal coordinates."""
        # Create branched structure: 0-1-2 with 3 bonded to 1
        mol_graph = nx.Graph()
        mol_graph.add_edges_from([(0, 1), (1, 2), (1, 3)])

        # Use first three atoms as cartesian reference (minimum needed)
        initial_cartesian = [0, 1, 2]

        descriptors, cartesian_indices = extract_descriptors(mol_graph, initial_cartesian)

        # Should have descriptors in (A, B, C, D) format
        assert descriptors.shape[0] == 4
        # No descriptors for branched case - atom 3 cannot be described with internal coordinates
        assert descriptors.shape[1] == 0

        # Should include all atoms in cartesian coordinates
        assert 0 in cartesian_indices
        assert 1 in cartesian_indices
        assert 2 in cartesian_indices
        assert 3 in cartesian_indices

    def test_empty_initial_cartesian(self):
        """Test with empty initial cartesian list."""
        # Create branched structure: 0-1-2-3 with 4 bonded to 2
        mol_graph = nx.Graph()
        mol_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 4)])

        # Use first three atoms as cartesian reference (minimum needed)

        descriptors, cartesian_indices = extract_descriptors(mol_graph, [])

        # Descriptors should be empty since no initial cartesian atoms provided
        assert descriptors.shape[0] == 4
        assert descriptors.shape[1] == 0

        # Should include all atoms in cartesian coordinates
        assert 0 in cartesian_indices
        assert 1 in cartesian_indices
        assert 2 in cartesian_indices
        assert 3 in cartesian_indices
        assert 4 in cartesian_indices

    def test_disconnected_graph(self):
        """Test with disconnected molecular graph."""
        mol_graph = nx.Graph()
        mol_graph.add_edges_from([(0, 1), (2, 3)])  # Two separate pairs

        initial_cartesian = [0]

        descriptors, cartesian_indices = extract_descriptors(mol_graph, initial_cartesian)

        # Descriptors should be empty
        assert descriptors.shape[0] == 4
        assert descriptors.shape[1] == 0

        # Disconnected atoms should be in cartesian coordinates
        assert 0 in cartesian_indices
        assert 1 in cartesian_indices
        assert 2 in cartesian_indices
        assert 3 in cartesian_indices


class TestComputeAngle:
    """Test the compute_angle function."""

    def test_right_angle(self):
        """Test computation of a 90-degree angle."""
        # Three points forming a right angle at B
        pos_a = np.array([[1.0, 0.0, 0.0]])
        pos_b = np.array([[0.0, 0.0, 0.0]])
        pos_c = np.array([[0.0, 1.0, 0.0]])

        out = np.zeros(1)
        compute_angle(out, pos_a, pos_b, pos_c)

        assert_allclose(out[0], np.pi/2, rtol=1e-7)

    def test_straight_angle(self):
        """Test computation of a 180-degree angle."""
        pos_a = np.array([[1.0, 0.0, 0.0]])
        pos_b = np.array([[0.0, 0.0, 0.0]])
        pos_c = np.array([[-1.0, 0.0, 0.0]])

        out = np.zeros(1)
        compute_angle(out, pos_a, pos_b, pos_c)

        assert_allclose(out[0], np.pi, rtol=1e-7)

    def test_zero_angle(self):
        """Test computation of a 0-degree angle."""
        pos_a = np.array([[1.0, 0.0, 0.0]])
        pos_b = np.array([[0.0, 0.0, 0.0]])
        pos_c = np.array([[1.0, 0.0, 0.0]])

        out = np.zeros(1)
        compute_angle(out, pos_a, pos_b, pos_c)

        assert_allclose(out[0], 0.0, atol=1e-7)

    def test_multiple_angles(self):
        """Test computation of multiple angles simultaneously."""
        pos_a = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        pos_b = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        pos_c = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

        out = np.zeros(2)
        compute_angle(out, pos_a, pos_b, pos_c)

        expected = np.array([np.pi/2, np.pi])
        assert_allclose(out, expected, rtol=1e-7)

    def test_degenerate_case(self):
        """Test with degenerate geometry (coincident points)."""
        pos_a = np.array([[0.0, 0.0, 0.0]])
        pos_b = np.array([[0.0, 0.0, 0.0]])  # Same as A
        pos_c = np.array([[1.0, 0.0, 0.0]])

        out = np.zeros(1)
        compute_angle(out, pos_a, pos_b, pos_c)

        # Should handle gracefully (result is 0)
        assert_allclose(out[0], 0.0, atol=1e-7)


class TestComputeDihedral:
    """Test the compute_dihedral function."""

    def test_planar_dihedral(self):
        """Test dihedral angle for planar configuration."""
        # Four points in a plane should give 0 or π dihedral
        pos_a = np.array([[0.0, 0.0, 0.0]])
        pos_b = np.array([[1.0, 0.0, 0.0]])
        pos_c = np.array([[2.0, 0.0, 0.0]])
        pos_d = np.array([[3.0, 0.0, 0.0]])

        out = np.zeros((2, 1))
        compute_dihedral(out, pos_a, pos_b, pos_c, pos_d)

        # For planar case, sine should be ~0
        assert_allclose(out[0, 0], 0.0, atol=1e-7)

    def test_perpendicular_dihedral(self):
        """Test dihedral angle for perpendicular planes."""
        pos_a = np.array([[0.0, 1.0, 0.0]])
        pos_b = np.array([[0.0, 0.0, 0.0]])
        pos_c = np.array([[1.0, 0.0, 0.0]])
        pos_d = np.array([[1.0, 0.0, 1.0]])

        out = np.zeros((2, 1))
        compute_dihedral(out, pos_a, pos_b, pos_c, pos_d)

        # Should give ±90 degree dihedral
        angle = np.arctan2(out[0, 0], out[1, 0])
        assert_allclose(abs(angle), np.pi/2, rtol=1e-7)

    def test_multiple_dihedrals(self):
        """Test computation of multiple dihedral angles simultaneously."""
        # First dihedral: planar (all atoms in xy-plane)
        # Second dihedral: non-planar (fourth atom out of plane)
        pos_a = np.array([
            [0.0, 0.0, 0.0],  # First dihedral A
            [0.0, 0.0, 0.0]   # Second dihedral A
        ])
        pos_b = np.array([
            [1.0, 0.0, 0.0],  # First dihedral B
            [1.0, 0.0, 0.0]   # Second dihedral B
        ])
        pos_c = np.array([
            [2.0, 0.0, 0.0],  # First dihedral C
            [2.0, 0.0, 0.0]   # Second dihedral C
        ])
        pos_d = np.array([
            [3.0, 0.0, 0.0],  # First dihedral D (planar)
            [2.0, 1.0, 0.0]   # Second dihedral D (non-planar)
        ])

        out = np.zeros((2, 2))
        compute_dihedral(out, pos_a, pos_b, pos_c, pos_d)

        # First dihedral should be planar (sine ~0)
        assert_allclose(out[0, 0], 0.0, atol=1e-7)

        # Second dihedral should have some reasonable values
        angle2 = np.arctan2(out[0, 1], out[1, 1])
        assert abs(angle2) <= np.pi  # Should be a valid angle

    def test_collinear_atoms(self):
        """Test with collinear atoms (degenerate case)."""
        pos_a = np.array([[0.0, 0.0, 0.0]])
        pos_b = np.array([[1.0, 0.0, 0.0]])
        pos_c = np.array([[2.0, 0.0, 0.0]])
        pos_d = np.array([[3.0, 0.0, 0.0]])  # All collinear

        out = np.zeros((2, 1))
        compute_dihedral(out, pos_a, pos_b, pos_c, pos_d)

        # Should handle gracefully
        assert out[0, 0] == 0.0
        assert out[1, 0] == 0.0


class TestComputeInternalCoordinates:
    """Test the compute_internal_coordinates function."""

    def test_simple_case(self):
        """Test internal coordinate computation for a simple molecule."""
        # Set up a simple 4-atom system
        positions = np.array([
            [0.0, 0.0, 0.0],  # atom 0
            [1.0, 0.0, 0.0],  # atom 1
            [2.0, 0.0, 0.0],  # atom 2
            [2.0, 1.0, 0.0],  # atom 3
        ])

        # Descriptors: place atom 3 using atoms 0, 1, 2
        descriptors = np.array([[0], [1], [2], [3]], dtype=np.int64)
        cartesian_indices = np.array([0, 1, 2])

        # Output arrays
        out_positions = np.zeros((3, 3))
        out_lengths = np.zeros(1)
        out_angles = np.zeros(1)
        out_dihedrals = np.zeros((2, 1))

        compute_internal_coordinates(
            positions, descriptors, cartesian_indices,
            out_positions, out_lengths, out_angles, out_dihedrals
        )

        # Check outputs
        assert_allclose(out_positions, positions[:3])
        assert_allclose(out_lengths[0], 1.0)  # Distance 2-3
        assert_allclose(out_angles[0], np.pi/2)  # Angle 1-2-3 is 90 degrees
        # Dihedral 0-1-2-3 should be planar (sine component ~0)
        assert_allclose(out_dihedrals[0, 0], 0.0, atol=1e-7)

    def test_multiple_descriptors(self):
        """Test with multiple internal coordinate descriptors simultaneously."""
        # Set up a 5-atom system with two internal coordinate descriptors
        positions = np.array([
            [0.0, 0.0, 0.0],  # atom 0
            [1.0, 0.0, 0.0],  # atom 1
            [2.0, 0.0, 0.0],  # atom 2
            [2.0, 1.0, 0.0],  # atom 3
            [3.0, 0.0, 0.0],  # atom 4
        ])

        # Two descriptors: atom 3 using atoms 0,1,2 and atom 4 using atoms 1,2,3
        descriptors = np.array([
            [0, 1],  # A atoms
            [1, 2],  # B atoms
            [2, 3],  # C atoms
            [3, 4],  # D atoms
        ], dtype=np.int64)
        cartesian_indices = np.array([0, 1, 2])

        # Output arrays for 2 descriptors
        out_positions = np.zeros((3, 3))
        out_lengths = np.zeros(2)
        out_angles = np.zeros(2)
        out_dihedrals = np.zeros((2, 2))

        compute_internal_coordinates(
            positions, descriptors, cartesian_indices,
            out_positions, out_lengths, out_angles, out_dihedrals
        )

        # Check outputs
        assert_allclose(out_positions, positions[:3])

        # First descriptor (0-1-2-3): Distance 2-3
        assert_allclose(out_lengths[0], 1.0)
        # Second descriptor (1-2-3-4): Distance 3-4
        assert_allclose(out_lengths[1], np.sqrt(2.0))  # Distance from (2,1,0) to (3,0,0)

        # First descriptor: Angle 1-2-3 is 90 degrees
        assert_allclose(out_angles[0], np.pi/2)
        # Second descriptor: Angle 2-3-4 is 45 degrees
        assert_allclose(out_angles[1], np.pi/4, rtol=1e-6)

        # First descriptor: Dihedral 0-1-2-3 should be planar (sine component ~0)
        assert_allclose(out_dihedrals[0, 0], 0.0, atol=1e-7)
        # Second descriptor: Dihedral 1-2-3-4 should also be planar (sine component ~0)
        assert_allclose(out_dihedrals[0, 1], 0.0, atol=1e-7)


if __name__ == "__main__":
    pytest.main([__file__])
