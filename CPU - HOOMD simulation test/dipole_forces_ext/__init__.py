"""dipole_forces_ext

This package bundles:
  - dipole_forces_ext._dipole_forces_ext: compiled real-space dipole force/energy kernel
  - dipole_forces_ext.ewald: Spectral-Ewald utilities (pure Python)
  - dipole_forces_ext.capacitance: mutual-polarization (capacitance) solver (pure Python)

The compiled kernel is intentionally kept independent of HOOMD internals; HOOMD v6
access happens through hoomd.md.force.Custom in the driver scripts.
"""

from __future__ import annotations

from ._dipole_forces_ext import compute_forces, __version__

__all__ = ["compute_forces", "__version__"]
