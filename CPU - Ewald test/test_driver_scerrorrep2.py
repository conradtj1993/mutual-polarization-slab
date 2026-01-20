"""
test_driver_scerrorrep2.py

Python translation of the MATLAB test driver provided by the user.

Notes:
  - Put this file in the same folder as capacitance.py and ewald.py.
  - In Spyder, set the working directory to this folder, then Run.
  - Outputs SCErrorREP2.mat with keys: 'surfs', 'betas', 'h'.

The only optional enhancement is "verbose" printouts to show progress.
"""

from __future__ import annotations

import numpy as np
from scipy.io import savemat

from capacitance import run_capacitance


def build_simple_cubic(numCells: int = 13) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Build the same simple-cubic lattice as in the MATLAB script.

    Returns
    -------
    x : (N,3) positions
    L0 : (3,) box lengths
    A  : yz-face area (L0[1]*L0[2])
    """
    a1 = np.array([2.0, 0.0, 0.0], dtype=np.float64)
    a2 = np.array([0.0, 2.0, 0.0], dtype=np.float64)
    a3 = np.array([0.0, 0.0, 2.0], dtype=np.float64)

    x_rows = []
    for a in range(numCells):
        for b in range(numCells):
            for c in range(numCells):
                x_rows.append(a * a1 + b * a2 + c * a3)

    x = np.asarray(x_rows, dtype=np.float64)
    L0 = numCells * np.array([np.linalg.norm(a1), np.linalg.norm(a2), np.linalg.norm(a3)], dtype=np.float64)
    A = float(L0[1] * L0[2])
    return x, L0, A


def main() -> None:
    betas = np.array([-0.5, -0.25, -0.05, 0.1, 0.5, 1.0], dtype=np.float64)
    betas=np.array([-0.01])
    h = np.concatenate(([0.0], np.logspace(-5.0, 1.0, 160, base=10.0))).astype(np.float64)
    h = np.array([0,100])
    H0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    nBeta = betas.size
    nH = h.size
    surfs = np.zeros((nBeta, nH), dtype=np.float64)

    x, L_0, A = build_simple_cubic(numCells=21)
    N = x.shape[0]

    xi = 0.5
    
    # Set this to 1,2,3 to print progress from the solver.
    verbose = 1

    for i, beta in enumerate(betas):
        eps_p = (1.0 + 2.0 * beta) / (1.0 - beta)

        C = np.zeros((nH,), dtype=np.float64)

        print(f"=== beta[{i+1}/{nBeta}] = {beta:+.3g}  eps_p={eps_p} ===", flush=True)

        for j, hj in enumerate(h):
            L = L_0 + np.array([0, 0.0, hj * L_0[0]], dtype=np.float64)
            Cj, _ = run_capacitance(x, L, eps_p * np.ones((N,), dtype=np.float64), xi, H0, verbose=verbose)
            C[j] = float(np.real(Cj[0]))

            if (j + 1) % 10 == 0:
                print(f"  h[{j+1}/{nH}] = {hj:.3e}  C={C[j]:.6g}", flush=True)

        U = -0.5 * C * float(N)
        u_s_h = (U - U[0]) / (2.0 * A)

        p_b = C[0] / np.sqrt(4.0 * np.pi)
        u_s_h = u_s_h / (p_b ** 2)

        surfs[i, :] = u_s_h
    print(surfs)

if __name__ == "__main__":
    main()
