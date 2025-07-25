SmoothMD
==============================

Tools for smoothing molecular dynamics trajectories using internal coordinates.

## Overview

SmoothMD is a Python library that provides trajectory smoothing capabilities for molecular dynamics (MD) simulations. The library applies temporal filtering to smooth MD trajectories while preserving the physical meaningfulness of the molecular structures.

### Key Features

- **Internal Coordinates Filtering**: Smoothing is performed in both cartesian and internal coordinates space (bond lengths, angles, and dihedral angles). This approach ensures that the resulting trajectory maintains realistic molecular geometries while reducing high-frequency noise commonly found in MD simulations.
- **Flexible Filtering**: Customizable filtering weights.

## Installation

To build SmoothMD from source,
we highly recommend using virtual environments.
Below we provide instructions for `pip`.

### Prerequisites

SmoothMD requires the following dependencies:
- Python 3.9+
- MDAnalysis
- NumPy
- NetworkX

### With pip

To build the package from source, run:

```
pip install .
```

If you want to create a development environment, install
the dependencies required for tests and docs with:

```
pip install ".[test]"
```

## Usage

### Basic Example

```python
import MDAnalysis as mda
from smoothmd import filter_smooth_trajectory

# Load your trajectory
universe = mda.Universe("topology.tpr", "trajectory.xtc")

# Select backbone atoms to be smoothed in Cartesian coordinates for better
# stability (recommended for polymers), except for prolines
# At least three atoms must be selected for each molecule in the system
cartesian_indices = universe.select_atoms("backbone and not resname PRO").ids

# Apply smoothing filter
filter_smooth_trajectory(
    "smooth_trajectory.xtc",  # output trajectory file
    universe,                 # input MDAnalysis universe
    cartesian_indices,        # atoms to be smoothed in Cartesian coordinates
    20,                       # period: filters-out movements with 20-frames long period
    5                         # step_frames: output every 5th frame
)
```

### Advanced Usage

#### Custom Filter Functions

You can define custom filter functions for different smoothing behaviors. Weights are computed using the provided function and are always normalized.

```python
import numpy as np
from smoothmd import filter_smooth_trajectory

def gaussian_filter(x):
    """Gaussian filter function"""
    return np.exp(-0.5 * (x * 4)**2)  # sigma = 0.25

def rectangular_filter(x):
    """Simple rectangular (uniform) filter"""
    return np.ones_like(x)

# Use custom filter
filter_smooth_trajectory(
    "smooth_trajectory.xtc",
    universe,
    cartesian_indices,
    period=15,
    filter_func=gaussian_filter
)
```

### Performance Tips

1. **Backbone Selection**: For proteins, filtering backbone atoms except for prolines in cartesian coordinates (`"backbone and not resname PRO"`) provides better results.
2. **Filter Window**: Value for `period` should be the period of the movements to be filtered in frames.
3. **Frame Skipping**: Use `skip_frames > 1` to reduce output trajectory size while maintaining smoothing quality.
4. **Memory Usage**: The algorithms use a streaming approach, so no additional memory is needed to store trajectories besides the one already allocated bu the `MDAnalysis.Universe`.

### Copyright

The SmoothMD source code is hosted at https://github.com/sciencecolors/smoothmd
and is available under the GNU Lesser General Public License, version 2.1 (see the file [LICENSE](https://github.com/sciencecolors/smoothmd/blob/main/LICENSE)).

Copyright (c) 2025, Caio S. Souza

## Citation

If you use SmoothMD in your research, please cite this software along with MDAnalysis:

```
SmoothMD: Tools for smoothing molecular dynamics trajectories
Caio S. Souza (2025)
https://github.com/sciencecolors/smoothmd
```

Also cite [MDAnalysis](https://github.com/MDAnalysis/mdanalysis#citation) as required.

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the GNU Lesser General Public License v2.1 - see the [LICENSE](LICENSE) file for details.

### Acknowledgements

Project based on the
[MDAnalysis Cookiecutter](https://github.com/MDAnalysis/cookiecutter-mda) version 0.1.
