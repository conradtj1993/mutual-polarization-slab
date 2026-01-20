"""
surface_energy.py

Convenience wrappers for surface energy calculations using run_capacitance().

This mirrors the logic in your MATLAB driver (surface_energy_coordsOnly.m /
SurfaceEnergyAnisotropy.m): compute capacitances at two separations and convert
into a surface energy per face.

This file is intentionally simple and function-based (no classes).
"""

from __future__ import annotations

import numpy as np

from capacitance import run_capacitance


def surface_energy_from_two_separations(
    xyz: np.ndarray,
    Lbox: np.ndarray,
    epsp: float,
    xi: float,
    Hhat: np.ndarray,
    hsep: np.ndarray,
    slab_factor: float = 1.0,
    verbose: int = 0,
) -> float:
    """
    Compute nondimensionalized surface energy from two capacitance calculations.

    Parameters
    ----------
    xyz : (N,3)
    Lbox : (3,)
    epsp : float (particle permittivity)
    xi : float
    Hhat : (3,) applied-field direction (unit)
    hsep : (2,) separations in multiples of Lbox[2] (like MATLAB example [0;2])
    verbose : int passed to run_capacitance

    Returns
    -------
    gamma_nd : float
        nondimensionalized surface energy uS / pB^2 (same as MATLAB)
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    Lbox = np.asarray(Lbox, dtype=np.float64).reshape(3)
    Hhat = np.asarray(Hhat, dtype=np.float64).reshape(3)
    hsep = np.asarray(hsep, dtype=np.float64).reshape(-1)
    if hsep.size != 2:
        raise ValueError("hsep must have exactly two entries (e.g., [0, 2]).")

    N = xyz.shape[0]
    A_face = float(Lbox[0] * Lbox[1])

    C = np.zeros((2,), dtype=np.float64)
    for j in range(2):
        Ltmp = Lbox.copy()
        Ltmp[2] = Ltmp[2] + hsep[j] * Lbox[2]
        Cj, _ = run_capacitance(xyz, Ltmp, epsp * np.ones((N,)), xi, Hhat, slab_factor=slab_factor, verbose=verbose)
        C[j] = float(np.real(Cj[0]))

    U = -0.5 * C * float(N)
    uS = (U[1] - U[0]) / (2.0 * A_face)
    pB = C[0] / np.sqrt(4.0 * np.pi)
    return float(uS / (pB ** 2))
