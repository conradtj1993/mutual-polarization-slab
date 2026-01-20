"""dipole_forces_ext.hoomd_force

HOOMD v6.x CustomForce helpers for the dipole_forces_ext package.

This module is optional and only intended to be imported in an environment where
HOOMD is installed.

It provides:
  - MutualPolarizationDipoleForceCached: solves induced dipoles via
    dipole_forces_ext.capacitance.run_capacitance, then computes real-space
    dipole forces/energies via dipole_forces_ext.compute_forces.

  - FixedDipoleForce: applies the real-space dipole kernel for fixed dipoles.

Design notes
------------
- Caching the dipole solve is deliberate: the GMRES solve (capacitance) is
  typically much more expensive than the real-space force evaluation.

- The slab correction, if enabled, is handled inside the dipole *solve* (via
  capacitance -> ewald.magnetic_field). The real-space force kernel can still be
  evaluated with full 3D periodicity (periodic_z=True), which is often the
  desired compromise when using a slab correction.

- The force kernel is independent of HOOMD internals: it only needs positions,
  dipoles, and box lengths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# HOOMD is an optional dependency for this module.
try:
    import hoomd
    import hoomd.md
except Exception as e:  # pragma: no cover
    raise ImportError(
        "dipole_forces_ext.hoomd_force requires HOOMD (v6.x). "
        "Import this module only inside a HOOMD-enabled environment."
    ) from e

from . import compute_forces
from . import capacitance


def _write_force_energy_arrays(arrays, f: np.ndarray, e: np.ndarray) -> None:
    """Write force + per-particle energy to HOOMD v6 force arrays.

    HOOMD's attribute name for energy is `potential_energy` for MD forces.
    Some older/beta builds may expose `energy`. This helper handles both.
    """

    arrays.force[:] = f

    if hasattr(arrays, "potential_energy"):
        arrays.potential_energy[:] = e
    elif hasattr(arrays, "energy"):
        arrays.energy[:] = e

    # The dipole kernel here does not define torques/virials.
    if hasattr(arrays, "torque"):
        arrays.torque[:] = 0.0
    if hasattr(arrays, "virial"):
        arrays.virial[:] = 0.0


@dataclass
class _CachedDipoles:
    by_tag: np.ndarray  # (max_tag+1,3)
    last_pol_step: Optional[int] = None


class MutualPolarizationDipoleForceCached(hoomd.md.force.Custom):
    """Mutual polarization (capacitance) + real-space dipole forces.

    This force solves for induced dipoles only every `pol_every_steps` and
    reuses the cached dipoles in between.

    Parameters
    ----------
    eps_p:
        Particle permittivity (scalar). Internally passed as an array of length N.
    xi, errtol, slab_factor, Hhat, use_scipy_gmres:
        Passed through to capacitance.run_capacitance.
    r_cut, prefactor:
        Real-space dipole kernel parameters passed to dipole_forces_ext.compute_forces.
    field_strength:
        Multiplies the unit-field dipoles returned by run_capacitance.
    periodic_z:
        If True, real-space dipole kernel uses minimum image in z.
    use_cell_list:
        If True, use internal cell list in the kernel (recommended for large N).
    pol_every_steps:
        GMRES solve cadence in steps. Must be >= 1.
    """

    def __init__(
        self,
        *,
        eps_p: float,
        xi: float,
        errtol: float,
        slab_factor: float,
        Hhat: np.ndarray,
        r_cut: float,
        prefactor: float,
        field_strength: float,
        pol_every_steps: int = 1,
        use_scipy_gmres: bool = True,
        periodic_z: bool = True,
        use_cell_list: bool = True,
    ):
        super().__init__()

        self.eps_p = float(eps_p)
        self.xi = float(xi)
        self.errtol = float(errtol)
        self.slab_factor = float(slab_factor)
        self.Hhat = np.asarray(Hhat, dtype=np.float64).reshape(3)

        self.r_cut = float(r_cut)
        self.prefactor = float(prefactor)
        self.field_strength = float(field_strength)

        self.use_scipy_gmres = bool(use_scipy_gmres)
        self.periodic_z = bool(periodic_z)
        self.use_cell_list = bool(use_cell_list)

        self.pol_every_steps = int(max(1, int(pol_every_steps)))

        # Diagnostics
        self.calls = 0
        self.pol_solves = 0
        self.last_dipoles: Optional[np.ndarray] = None
        self.last_forces: Optional[np.ndarray] = None
        self.last_energy: Optional[np.ndarray] = None

        self._cache: Optional[_CachedDipoles] = None

    def _ensure_cache(self, tags: np.ndarray) -> None:
        if self._cache is None:
            max_tag = int(np.max(tags)) if tags.size else -1
            self._cache = _CachedDipoles(by_tag=np.zeros((max_tag + 1, 3), dtype=np.float64))
        else:
            # Expand if needed (should be rare).
            max_tag = int(np.max(tags)) if tags.size else -1
            if max_tag >= self._cache.by_tag.shape[0]:
                new = np.zeros((max_tag + 1, 3), dtype=np.float64)
                new[: self._cache.by_tag.shape[0], :] = self._cache.by_tag
                self._cache.by_tag = new

    def _solve_dipoles(self, pos: np.ndarray, boxL: np.ndarray, tags: np.ndarray) -> None:
        """Solve induced dipoles for unit field along Hhat, then scale by field_strength."""
        Nloc = int(pos.shape[0])
        eps = np.full(Nloc, self.eps_p, dtype=np.float64)

        _C, p = capacitance.run_capacitance(
            pos,
            boxL,
            eps,
            float(self.xi),
            self.Hhat.copy(),
            errortol=float(self.errtol),
            slab_factor=float(self.slab_factor),
            verbose=0,
            use_scipy_gmres=bool(self.use_scipy_gmres),
        )

        p_arr = np.asarray(p)
        if p_arr.ndim == 3:
            p0 = np.real(p_arr[:, :, 0]).astype(np.float64, copy=False)
        elif p_arr.ndim == 2:
            p0 = np.real(p_arr).astype(np.float64, copy=False)
        else:
            raise RuntimeError(f"Unexpected p shape from run_capacitance: {p_arr.shape}")

        dip_loc = (p0 * self.field_strength).astype(np.float64, copy=False)

        self._ensure_cache(tags)
        assert self._cache is not None

        # Cache by tag
        for i in range(Nloc):
            self._cache.by_tag[int(tags[i])] = dip_loc[i]

        self.pol_solves += 1

    def set_forces(self, timestep):
        self.calls += 1

        with self._state.cpu_local_snapshot as snap:
            pos = np.asarray(snap.particles.position, dtype=np.float64).copy()
            tags = np.asarray(snap.particles.tag, dtype=np.int64).copy()
            boxL = np.asarray(self._simulation.state.box.L, dtype=np.float64).reshape(3).copy()

        if pos.shape[0] == 0:
            return

        self._ensure_cache(tags)
        assert self._cache is not None

        do_solve = False
        if self._cache.last_pol_step is None:
            do_solve = True
        elif (int(timestep) - int(self._cache.last_pol_step)) >= int(self.pol_every_steps):
            do_solve = True

        if do_solve:
            self._solve_dipoles(pos, boxL, tags)
            self._cache.last_pol_step = int(timestep)

        dip = self._cache.by_tag[tags].astype(np.float64, copy=False)

        f, e = compute_forces(
            pos.astype(np.float64, copy=False),
            dip.astype(np.float64, copy=False),
            boxL.astype(np.float64, copy=False),
            float(self.r_cut),
            float(self.prefactor),
            periodic_z=bool(self.periodic_z),
            use_cell_list=bool(self.use_cell_list),
        )

        f = np.asarray(f, dtype=np.float64)
        e = np.asarray(e, dtype=np.float64).reshape(-1)

        if f.shape != (pos.shape[0], 3):
            raise RuntimeError(f"Force shape mismatch: got {f.shape}, expected {(pos.shape[0], 3)}")
        if e.shape != (pos.shape[0],):
            raise RuntimeError(f"Energy shape mismatch: got {e.shape}, expected {(pos.shape[0],)}")
        if np.any(~np.isfinite(f)) or np.any(~np.isfinite(e)) or np.any(~np.isfinite(dip)):
            raise RuntimeError("Non-finite values in dipoles/forces/energies")

        with self.cpu_local_force_arrays as arrays:
            _write_force_energy_arrays(arrays, f, e)

        self.last_dipoles = dip.copy()
        self.last_forces = f.copy()
        self.last_energy = e.copy()


class FixedDipoleForce(hoomd.md.force.Custom):
    """Real-space dipole kernel for fixed dipoles mu*Hhat."""

    def __init__(
        self,
        *,
        Hhat: np.ndarray,
        mu: float,
        r_cut: float,
        prefactor: float,
        periodic_z: bool = True,
        use_cell_list: bool = True,
    ):
        super().__init__()
        self.Hhat = np.asarray(Hhat, dtype=np.float64).reshape(3)
        self.mu = float(mu)
        self.r_cut = float(r_cut)
        self.prefactor = float(prefactor)
        self.periodic_z = bool(periodic_z)
        self.use_cell_list = bool(use_cell_list)

        self.calls = 0
        self.last_forces: Optional[np.ndarray] = None
        self.last_energy: Optional[np.ndarray] = None

    def set_forces(self, timestep):
        self.calls += 1

        with self._state.cpu_local_snapshot as snap:
            pos = np.asarray(snap.particles.position, dtype=np.float64).copy()
            boxL = np.asarray(self._simulation.state.box.L, dtype=np.float64).reshape(3).copy()

        Nloc = int(pos.shape[0])
        if Nloc == 0:
            return

        dip = (self.mu * self.Hhat).reshape(1, 3).repeat(Nloc, axis=0).astype(np.float64, copy=False)

        f, e = compute_forces(
            pos.astype(np.float64, copy=False),
            dip.astype(np.float64, copy=False),
            boxL.astype(np.float64, copy=False),
            float(self.r_cut),
            float(self.prefactor),
            periodic_z=bool(self.periodic_z),
            use_cell_list=bool(self.use_cell_list),
        )

        f = np.asarray(f, dtype=np.float64)
        e = np.asarray(e, dtype=np.float64).reshape(-1)

        with self.cpu_local_force_arrays as arrays:
            _write_force_energy_arrays(arrays, f, e)

        self.last_forces = f.copy()
        self.last_energy = e.copy()
