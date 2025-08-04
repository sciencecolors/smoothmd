"""
Unit and regression test for the smoothmd package.
"""

import sys
import smoothmd


def test_smoothmd_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "smoothmd" in sys.modules


def test_package_version():
    """Test that package has a version."""
    assert hasattr(smoothmd, "__version__")
