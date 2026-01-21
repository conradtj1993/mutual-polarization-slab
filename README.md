# mutual-polarization

Computational physics library for dipolar interaction modeling in slab geometries. Implements a mutual polarization solver (capacitance method) with spectral-Ewald acceleration for long-range dipole-dipole interactions.

## Features

- **Mutual polarization solver** - Self-consistent induced dipole calculation via GMRES
- **Spectral-Ewald summation** - Efficient long-range dipole-dipole interactions
- **Slab geometry support** - Proper handling of confined (non-periodic z) systems
- **HOOMD v6 integration** - Custom force wrappers for molecular dynamics simulations
- **C++ accelerated** - Real-space kernel with cell-list neighbor finding

## Installation

### Requirements

- Python 3.7+
- NumPy
- pybind11
- C++17 compatible compiler
- SciPy (optional, improves GMRES performance)
- HOOMD v6 (optional, for MD simulations)

### Building the C++ Extension

```bash
cd "CPU - HOOMD simulation test"
python setup_dipole_forces_ext.py build_ext --inplace
```

## Usage

### Pure Python (Reference Implementation)

```python
from capacitance import run_capacitance

# Solve for induced dipoles
dipoles = run_capacitance(positions, box, eps_p, xi, H_applied, ...)
```

### HOOMD v6 Integration

```python
from dipole_forces_ext.hoomd_force import MutualPolarizationDipoleForceCached

# Create force object that solves mutual polarization every N steps
force = MutualPolarizationDipoleForceCached(eps_p, xi, r_cut, pol_every_steps=10)
```

## Directory Structure

| Directory | Description |
|-----------|-------------|
| `CPU - Ewald test/` | Pure Python reference implementation and validation tests |
| `CPU - HOOMD simulation test/` | Production code with C++ extension and HOOMD integration |
| `Capacitance_*/` | MATLAB reference implementations |
| `Compute_Field_*/` | MATLAB field computation routines |

## Running Tests

```bash
# Pure Python validation (no HOOMD required)
cd "CPU - Ewald test"
python test_driver_scerrorrep2.py
python test_driver_BCT110.py

# HOOMD integration diagnostics
cd "CPU - HOOMD simulation test"
python simulation_diagnostic.py
```

## License

MIT License - see [LICENSE](LICENSE) for details.