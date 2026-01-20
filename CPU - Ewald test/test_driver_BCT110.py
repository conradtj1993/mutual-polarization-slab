
"""
test_driver_bct110.py

Modification of the user's working diagnostic driver to build a BCT(110) slab
configuration (as in the provided MATLAB snippet), then call run_capacitance.

IMPORTANT (axis convention):
  The underlying Ewald/slab correction in MagneticField assumes the "slab/non-periodic"
  direction is the *3rd* coordinate axis (z). If you want a slab whose *physical* normal
  is the x-axis (i.e., yz face area), you must either:
    (A) rotate/permutate coordinates so that physical x -> solver z, OR
    (B) use a solver that supports an arbitrary slab-normal.

This driver follows (A), using the same trick you used to fix the SC case:
  - We build the lattice in a natural "physical" coordinate system where the slab
    normal is x and the face area is yz.
  - Then we permute coordinates and box lengths so that the slab normal becomes z
    for the solver.

Permutation used:
  (x_phys, y_phys, z_phys) -> (x_sol, y_sol, z_sol) = (z_phys, y_phys, x_phys)

With this mapping:
  - The slab normal (physical x) becomes solver z.
  - A yz face area in physical coords becomes an xy face area in solver coords.
  - If your physical applied field is along physical z, then in solver coords it is along x,
    so H0 should be [1, 0, 0] (same as your working SC fix).
"""

from __future__ import annotations

import numpy as np
from scipy.io import savemat

from capacitance import run_capacitance


def build_bct_110_slab(numCells: int = 8, permute_for_solver: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Build the BCT(110) lattice/slab as in the provided MATLAB snippet:

        numCells = 8
        a1 = [sqrt(12), 0, 0]
        a2 = [0, sqrt(12), 0]
        a3 = [0, 0, 2]
        J = [ (0,0,0),
              (0,0.5,0.5),
              (0.5,0,0.5),
              (0.5,0.5,0) ]

        x = sum over basis J and cell indices a,b,c in [0, numCells-1]

        L0 = [numCells*|a1|, numCells*|a2|, numCells*|a3|]
        A  = L0_y * L0_z   (yz face area in the *physical* coordinates)

    Parameters
    ----------
    numCells : int
        Number of unit cells along each lattice vector.
    permute_for_solver : bool
        If True, permute axes so that the slab normal (physical x) maps to solver z.

    Returns
    -------
    x : (N,3) float64
        Particle positions (in solver coordinates if permute_for_solver=True).
    L0 : (3,) float64
        Box lengths (in solver coordinates if permute_for_solver=True).
    A : float
        Surface face area corresponding to the slab faces (two faces). This is
        the physical yz area; after permutation it equals solver x*y area.
    """
    a1 = np.array([np.sqrt(12.0), 0.0, 0.0], dtype=np.float64)
    a2 = np.array([0.0, np.sqrt(12.0), 0.0], dtype=np.float64)
    a3 = np.array([0.0, 0.0, 2.0], dtype=np.float64)

    J = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )

    x_rows: list[np.ndarray] = []
    for ji in J:  # ji is a length-3 row
        for a in range(numCells):
            for b in range(numCells):
                for c in range(numCells):
                    x_rows.append((a + ji[0]) * a1 + (b + ji[1]) * a2 + (c + ji[2]) * a3)

    x_phys = np.asarray(x_rows, dtype=np.float64)

    # Physical box lengths (as in MATLAB snippet)
    L0_phys = numCells * np.array(
        [np.linalg.norm(a1), np.linalg.norm(a2), np.linalg.norm(a3)],
        dtype=np.float64,
    )

    # Physical yz-face area
    A_phys = float(L0_phys[1] * L0_phys[2])

    if not permute_for_solver:
        return x_phys, L0_phys, A_phys

    # Permute axes so physical x becomes solver z:
    # (x_sol, y_sol, z_sol) = (z_phys, y_phys, x_phys)
    x_sol = x_phys[:, [2, 1, 0]].copy()
    L0_sol = np.array([L0_phys[2], L0_phys[1], L0_phys[0]], dtype=np.float64)

    # In solver coords, faces normal to solver z have area Lx*Ly = L0_sol[0]*L0_sol[1],
    # which equals A_phys by construction.
    return x_sol, L0_sol, A_phys


def main() -> None:
    # --- user settings (keep this block simple) ---
    betas = np.array([-0.01], dtype=np.float64)

    # diagnostic two-point "h": 0 and very large
    h = np.array([0.0, 100.0], dtype=np.float64)

    # After the axis permutation described above:
    # - physical field along physical z becomes solver x
    H0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    xi = 0.5
    verbose = 1  # 0 silent, 1 summary, 2+ more

    # --- build lattice ---
    x, L_0, A = build_bct_110_slab(numCells=8, permute_for_solver=True)
    N = x.shape[0]

    nBeta = betas.size
    nH = h.size
    surfs = np.zeros((nBeta, nH), dtype=np.float64)

    for i, beta in enumerate(betas):
        eps_p = (1.0 + 2.0 * beta) / (1.0 - beta)

        C = np.zeros((nH,), dtype=np.float64)

        print(f"=== beta[{i+1}/{nBeta}] = {beta:+.6g}  eps_p={eps_p:.6g}  N={N} ===", flush=True)
        print(f"L0 (solver coords) = {L_0}   A(face) = {A}", flush=True)

        for j, hj in enumerate(h):
            # We want to add vacuum along the slab-normal direction.
            # In solver coords, slab normal is z, so stretch Lz.
            L = L_0.copy()
            L[2] = L_0[2] + hj * L_0[2]  # same form as MATLAB: L = L0 + h*L0 along normal

            Cj, _ = run_capacitance(x, L, eps_p * np.ones((N,), dtype=np.float64), xi, H0, verbose=verbose)
            C[j] = float(np.real(Cj[0]))

            print(f"  h[{j+1}/{nH}] = {hj:.6g}  Lz={L[2]:.6g}  C={C[j]:.12g}", flush=True)

        # --- same postprocessing as MATLAB drivers ---
        U = -0.5 * C * float(N)
        u_s_h = (U - U[0]) / (2.0 * A)

        p_b = C[0] / np.sqrt(4.0 * np.pi)
        u_s_h = u_s_h / (p_b ** 2)

        surfs[i, :] = u_s_h

    print("surfs =", surfs, flush=True)

if __name__ == "__main__":
    main()

