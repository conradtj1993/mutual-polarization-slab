dipole_forces_ext (HOOMD v6)

Summary
- CPU implementation of real-space dipole–dipole forces/energies (compiled extension).
- Mutual polarization solver (“capacitance”) + spectral-Ewald utilities (“ewald”) (pure Python).
- HOOMD v6 integration via hoomd.md.force.Custom wrappers.

Package contents
- dipole_forces_ext/_dipole_forces_ext.*  (compiled via pybind11)
    compute_forces(pos, dip, box, r_cut, prefactor=1.0, periodic_z=True, use_cell_list=True)
      pos: (N,3) float64 positions
      dip: (N,3) float64 dipole vectors
      box: (3,) float64 box lengths [Lx, Ly, Lz]
      returns: (forces (N,3), potential_energy (N,))
- dipole_forces_ext/ewald.py
    Spectral-Ewald helper routines used by capacitance.py
- dipole_forces_ext/capacitance.py
    Induced dipole solver using GMRES (SciPy optional, otherwise internal restarted GMRES)
- dipole_forces_ext/hoomd_force.py
    FixedDipoleForce: uses compute_forces with fixed dipoles
    MutualPolarizationDipoleForceCached: solves induced dipoles every pol_every_steps and caches dipoles

Build / install (CPU)
Requirements
- Python (same version as your HOOMD build)
- numpy
- pybind11
- a C++17 compiler (g++/clang++)
Optional
- OpenMP (used if available; gives CPU parallelism)
- SciPy (used for scipy.sparse.linalg.gmres if enabled)

In-place build (development workflow)
- Put setup_dipole_forces_ext.py next to the package directory or scripts.
- Run:
    python setup_dipole_forces_ext.py build_ext --inplace

Editable install (if you have a proper package directory)
- From the project root:
    pip install -e .

GPU port (what would be required)
- Implement a HOOMD v6 C++ ForceCompute with a CUDA backend (NVCC build) that computes the
  dipole kernel on the GPU using HOOMD neighbor lists on-device.
- If you want the mutual polarization solve on GPU, capacitance/ewald must move off Python/NumPy
  (and likely require GPU FFT + GPU linear algebra / Krylov solver).
