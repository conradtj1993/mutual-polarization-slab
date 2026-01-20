"""
capacitance.py

Flat (MATLAB-like) Python consolidation of:
  - RunCapacitance.m
  - Capacitance.m
  - MagneticDipole.m

The goal is simplicity: drop these .py files in a folder (like MATLAB .m files),
and call run_capacitance(...) directly.

Performance note:
  The MATLAB workflow builds a *candidate* neighbor list from cell adjacency,
  then applies the real-space cutoff (d < rc) inside RealSpace.m. In Python,
  looping over all candidate pairs inside every GMRES matvec can be very slow.

  This version precomputes and filters the real-space pairs ONCE per Capacitance
  call, and then evaluates the real-space term with vectorized NumPy + scatter-add.
  This is mathematically identical to the MATLAB code (same cutoff rule, same
  interpolation), just reorganized for speed.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import time

try:
    # Prefer SciPy's tested Krylov solvers when available.
    from scipy.sparse.linalg import LinearOperator, gmres as scipy_gmres  # type: ignore
except Exception:  # pragma: no cover
    LinearOperator = None  # type: ignore
    scipy_gmres = None  # type: ignore

from .ewald import (
    pre_calculations,
    real_space_table,
    cell_list,
    neighbor_list,
    prepare_real_space_pairs,
    prepare_grid_kernel,
    precompute_kspace,
    magnetic_field,
)




# -----------------------------------------------------------------------------
# Lightweight caches (safe: results depend only on inputs)
# -----------------------------------------------------------------------------
# Real-space tabulation depends only on (xi) given the fixed r_table used here.
_REALSPACE_TABLE_CACHE: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

# -----------------------------------------------------------------------------
# Restarted GMRES (matrix-free), MATLAB-style controls.
# -----------------------------------------------------------------------------
def gmres_restarted(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    x0: np.ndarray,
    tol: float,
    restart: int,
    maxit: int,
    verbose: int = 0,
) -> Tuple[np.ndarray, int, float]:
    """
    Restarted GMRES solving A x = b using a matrix-free matvec.

    This is designed to match MATLAB's call signature:

        gmres(@fun, b, restart, tol, maxit, [], [], x0)

    Returns
    -------
    x : solution vector
    flag : 0 if converged, 1 otherwise
    relres : final relative residual ||b - A x|| / ||b||
    """
    b = np.asarray(b, dtype=np.complex128).reshape(-1)
    x = np.asarray(x0, dtype=np.complex128).reshape(-1).copy()

    nb = np.linalg.norm(b)
    if nb == 0.0:
        return x, 0, 0.0

    def mv(v: np.ndarray) -> np.ndarray:
        return np.asarray(matvec(v), dtype=np.complex128).reshape(-1)

    t0 = time.perf_counter()
    matvec_calls = 0

    # Outer (restart) loop
    for outer in range(maxit):
        r = b - mv(x)
        matvec_calls += 1
        beta = np.linalg.norm(r)
        relres = float(beta / nb)

        if verbose >= 2:
            print(f"    GMRES outer {outer+1}/{maxit}: relres={relres:.3e}", flush=True)

        if relres <= tol:
            if verbose >= 1:
                dt = time.perf_counter() - t0
                print(f"    GMRES converged: relres={relres:.3e}  outer={outer+1}  matvecs={matvec_calls}  time={dt:.2f}s", flush=True)
            return x, 0, relres

        # Arnoldi basis V and Hessenberg H (dimension restart+1 by restart)
        V = np.zeros((b.size, restart + 1), dtype=np.complex128)
        H = np.zeros((restart + 1, restart), dtype=np.complex128)

        V[:, 0] = r / beta

        # Right-hand side in Krylov basis
        g = np.zeros((restart + 1,), dtype=np.complex128)
        g[0] = beta

        # Givens rotations (stored as cos/sin-like scalars)
        cs = np.zeros((restart,), dtype=np.complex128)
        sn = np.zeros((restart,), dtype=np.complex128)

        inner_converged = False
        y_best = None
        relres_best = relres

        for j in range(restart):
            w = mv(V[:, j])
            matvec_calls += 1

            # Modified Gram-Schmidt
            for i in range(j + 1):
                H[i, j] = np.vdot(V[:, i], w)
                w = w - H[i, j] * V[:, i]

            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] != 0.0:
                V[:, j + 1] = w / H[j + 1, j]
            else:
                V[:, j + 1] = 0.0

            # Apply previous Givens rotations
            for i in range(j):
                tmp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -np.conj(sn[i]) * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = tmp

            # Compute new Givens rotation
            a = H[j, j]
            b2 = H[j + 1, j]
            denom = np.sqrt(np.abs(a) ** 2 + np.abs(b2) ** 2)
            if denom == 0.0:
                cs[j] = 1.0
                sn[j] = 0.0
            else:
                cs[j] = a / denom
                sn[j] = b2 / denom

            # Apply rotation to Hessenberg
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0

            # Apply rotation to g
            g_j = g[j]
            g[j] = cs[j] * g_j + sn[j] * g[j + 1]
            g[j + 1] = -np.conj(sn[j]) * g_j + cs[j] * g[j + 1]

            relres_inner = float(np.abs(g[j + 1]) / nb)
            if verbose >= 3:
                print(f"      inner {j+1}/{restart}: relres~{relres_inner:.3e}", flush=True)

            if relres_inner < relres_best:
                # Compute y (least-squares) for best-so-far
                y = np.linalg.lstsq(H[: j + 1, : j + 1], g[: j + 1], rcond=None)[0]
                y_best = y
                relres_best = relres_inner

            if relres_inner <= tol:
                # Solve least squares and update x
                y = np.linalg.lstsq(H[: j + 1, : j + 1], g[: j + 1], rcond=None)[0]
                x = x + V[:, : j + 1] @ y
                inner_converged = True
                break

        if not inner_converged:
            # Update using best available y (or last iterate if none)
            if y_best is None:
                y_best = np.linalg.lstsq(H[:restart, :restart], g[:restart], rcond=None)[0]
            x = x + V[:, : y_best.size] @ y_best

    # If we get here, we didn't converge
    r = b - mv(x)
    relres = float(np.linalg.norm(r) / nb)
    if verbose >= 1:
        dt = time.perf_counter() - t0
        print(f"    GMRES NOT converged: relres={relres:.3e}  outer={maxit}  matvecs={matvec_calls}  time={dt:.2f}s", flush=True)
    return x, 1, relres


# -----------------------------------------------------------------------------
# MagneticDipole.m
# -----------------------------------------------------------------------------
def magnetic_dipole(
    x: np.ndarray,
    lambdap: np.ndarray,
    H: np.ndarray,
    box: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    Ngrid: np.ndarray,
    h: np.ndarray,
    P: int,
    xi: float,
    eta: np.ndarray,
    rc: float,
    mold: np.ndarray,
    offset: np.ndarray,
    offsetxyz: np.ndarray,
    Hperp: np.ndarray,
    Hpara: np.ndarray,
    rvals: np.ndarray,
    errortol: float,
    real_precomp=None,
    grid_precomp=None,
    k_precomp=None,
    slab_factor: float = 1.0,
    use_scipy_gmres: bool = True,
    verbose: int = 0,
) -> np.ndarray:
    """
    Direct translation of MagneticDipole.m, but uses gmres_restarted above.

    real_precomp:
      If not None, this should be the tuple returned by prepare_real_space_pairs(),
      and magnetic_field() will use the fast real-space path.
    """
    N = x.shape[0]

    H = np.asarray(H, dtype=np.complex128).reshape(3)
    Hrep = np.tile(H, N)  # 3N vector

    mold_vec = np.asarray(mold, dtype=np.complex128).reshape((N, 3)).ravel()

    def fun(mprime: np.ndarray) -> np.ndarray:
        mmat = np.asarray(mprime, dtype=np.complex128).reshape((N, 3))
        Hprime = magnetic_field(
            x, mmat, lambdap, box, p1, p2,
            Ngrid, h, P, xi, eta, rc,
            offset, offsetxyz, Hperp, Hpara, rvals,
            real_precomp=real_precomp,
            grid_precomp=grid_precomp,
            k_precomp=k_precomp,
            slab_factor=slab_factor,
        )
        return Hprime.reshape(-1)

    restart = min(3 * N, 10)
    maxit = min(3 * N, 100)

    if verbose >= 2:
        print(f"  MagneticDipole: N={N} restart={restart} maxit={maxit} tol={errortol:g}", flush=True)

    # --- Solve A m = Hrep ---
    # MATLAB uses gmres(@fun, Hrep, restart, errortol, maxit, [], [], mold)
    # We prefer SciPy's gmres when available (more battle-tested than our small
    # pure-Python implementation), but keep the fallback for environments
    # without SciPy.
    if use_scipy_gmres and scipy_gmres is not None and LinearOperator is not None:
        Aop = LinearOperator((3 * N, 3 * N), matvec=fun, dtype=np.complex128)

        # Optional progress callback
        cb = None
        if verbose >= 3:
            def _cb(rk):
                # SciPy passes either the residual vector or its norm depending on version
                try:
                    val = float(rk)
                except Exception:
                    val = float(np.linalg.norm(rk))
                print(f"      SciPy GMRES: relres~{val:.3e}", flush=True)
            cb = _cb

        # SciPy API changed from tol -> (rtol, atol). Support both.
        try:
            sol, info = scipy_gmres(
                Aop, Hrep, x0=mold_vec,
                restart=restart, maxiter=maxit,
                rtol=errortol, atol=0.0,
                callback=cb,
            )
        except TypeError:
            sol, info = scipy_gmres(
                Aop, Hrep, x0=mold_vec,
                restart=restart, maxiter=maxit,
                tol=errortol,
                callback=cb,
            )

        # Compute *actual* relative residual (do not trust callback/estimates)
        r = Hrep - fun(sol)
        relres = float(np.linalg.norm(r) / np.linalg.norm(Hrep))
        flag = 0 if info == 0 and relres <= errortol else 1

        if verbose >= 1:
            print(f"  MagneticDipole(SciPy): info={info} relres={relres:.3e}", flush=True)

        return np.asarray(sol, dtype=np.complex128).reshape((N, 3))

    # Fallback: pure-Python restarted GMRES
    sol, flag, relres = gmres_restarted(fun, Hrep, mold_vec, errortol, restart, maxit, verbose=max(verbose - 1, 0))

    # Recompute actual residual once per solve for sanity (cheap compared to GMRES)
    r = Hrep - fun(sol)
    relres_actual = float(np.linalg.norm(r) / np.linalg.norm(Hrep))

    if verbose >= 1:
        print(f"  MagneticDipole(Python): flag={flag} relres_est={relres:.3e} relres={relres_actual:.3e}", flush=True)

    return sol.reshape((N, 3))


# -----------------------------------------------------------------------------
# Capacitance.m
# -----------------------------------------------------------------------------
def capacitance(
    x: np.ndarray,
    lambdap: np.ndarray,
    box: np.ndarray,
    Sguess: np.ndarray,
    xi: float,
    errortol: float,
    dipoletable1: np.ndarray,
    dipoletable2: np.ndarray,
    rtable: np.ndarray,
    offset: np.ndarray,
    offsetxyz: np.ndarray,
    Hhat: np.ndarray,
    slab_factor: float = 1.0,
    use_scipy_gmres: bool = True,
    verbose: int = 0,
) -> Tuple[float, np.ndarray]:
    """
    Self-consistent determination of induced dipoles and net capacitance.

    Direct translation of Capacitance.m, with one performance-preserving change:
    we precompute the filtered real-space pairs (d < rc) once, then reuse them
    inside GMRES matvecs.
    """
    x = np.asarray(x, dtype=np.float64)
    lambdap = np.asarray(lambdap, dtype=np.float64).reshape(-1)
    box = np.asarray(box, dtype=np.float64).reshape(3)
    Hhat = np.asarray(Hhat, dtype=np.float64).reshape(3)

    # --- spectral-Ewald parameters (exactly as MATLAB) ---
    rc = float(np.sqrt(-np.log(errortol)) / xi)
    kcut = 2.0 * xi * np.sqrt(-np.log(errortol))
    Ngrid = np.ceil(1.0 + box * kcut / np.pi).astype(int)
    h = box / Ngrid
    P = int(np.ceil(-2.0 * np.log(errortol) / np.pi))
    eta = P * (h * xi) ** 2 / np.pi  # spectral splitting parameter (vector length 3)

    if np.any(rc > box / 2.0):
        raise ValueError("Real-space cutoff exceeds half the box size; tolerance unmet.")

    t0 = time.perf_counter()

    # --- neighbour list (same as MATLAB) ---
    cell, Ncell = cell_list(x, box, rc)
    p1, p2 = neighbor_list(cell, Ncell)

    # Pre-filter + precompute interpolation geometry for RealSpace (fast path)
    t_rs0 = time.perf_counter()
    p1f, p2f, rhat, A, B = prepare_real_space_pairs(x, box, p1, p2, rc, dipoletable1, dipoletable2, rtable)
    real_precomp = (p1f, p2f, rhat, A, B)
    t_rs = time.perf_counter() - t_rs0

    # Precompute grid and reciprocal-space kernel data once per solve.
    t_g0 = time.perf_counter()
    grid_precomp = prepare_grid_kernel(x, Ngrid, h, xi, eta, P, offset, offsetxyz)
    t_g = time.perf_counter() - t_g0

    t_k0 = time.perf_counter()
    k_precomp = precompute_kspace(Ngrid, box, xi, eta)
    t_k = time.perf_counter() - t_k0

    if verbose >= 1:
        print(
            f"Capacitance: N={x.shape[0]} rc={rc:.3f} Ngrid={tuple(Ngrid.tolist())} P={P} "
            f"Ncell={tuple(Ncell.tolist())} pairs(cand)={p1.size} pairs(kept)={p1f.size} "
            f"prep_real={t_rs:.2f}s prep_grid={t_g:.2f}s prep_k={t_k:.2f}s",
            flush=True,
        )

    # --- solve dipoles with GMRES / Spectral Ewald ---
    S = magnetic_dipole(
        x, lambdap, Hhat, box, p1, p2,
        Ngrid, h, P, xi, eta, rc,
        Sguess, offset, offsetxyz,
        dipoletable1, dipoletable2, rtable,
        errortol,
        real_precomp=real_precomp,
        grid_precomp=grid_precomp,
        k_precomp=k_precomp,
        slab_factor=slab_factor,
        use_scipy_gmres=use_scipy_gmres,
        verbose=verbose,
    )

    # --- capacitance: average dipole component along the applied field ---
    C = np.mean(S @ Hhat)

    if verbose >= 1:
        dt = time.perf_counter() - t0
        print(f"Capacitance: done C={np.real(C):.6g}  total_time={dt:.2f}s", flush=True)

    return C, S


# -----------------------------------------------------------------------------
# RunCapacitance.m
# -----------------------------------------------------------------------------
def run_capacitance(
    x: np.ndarray,
    box: np.ndarray,
    eps_p: np.ndarray,
    xi: float,
    H0: np.ndarray,
    errortol: float = 1e-5,
    slab_factor: float = 1.0,
    use_scipy_gmres: bool = True,
    verbose: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute effective capacitance C and induced particle dipoles p for a configuration
    (or trajectory) of polarizable particles.

    Direct translation of RunCapacitance.m.

    Parameters
    ----------
    x : (N,3) or (N,3,Nframes)
    box : (3,)
    eps_p : (N,)
    xi : float
    H0 : (3,)
    errortol : float
    verbose : int
      0 silent; 1 per-call summaries; 2 includes GMRES outer residuals; 3 includes inner residuals.

    Returns
    -------
    C : (Nframes,) float array
    p : (N,3,Nframes) complex array
    """
    x = np.asarray(x, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64).reshape(3)
    eps_p = np.asarray(eps_p, dtype=np.float64).reshape(-1)

    if x.ndim == 2:
        x = x[:, :, np.newaxis]  # N×3×1

    N = x.shape[0]
    Nframes = x.shape[2]
    eta_vol = 4.0 * np.pi * N / (3.0 * np.prod(box))  # volume fraction, matches MATLAB

    # --- pre-tabulate real-space coefficients (exactly as MATLAB) ---
    # This table depends only on xi for the fixed r_table used here, so we cache it.
    xi_key = float(xi)
    if xi_key in _REALSPACE_TABLE_CACHE:
        field_dip_1, field_dip_2, r_table = _REALSPACE_TABLE_CACHE[xi_key]
    else:
        r_table = np.arange(0.001, 10.0 + 1e-12, 0.001, dtype=np.float64)
        _, _, field_dip_1, field_dip_2, _, _ = real_space_table(r_table, xi)
        r_table = np.concatenate(([0.0], r_table))
        _REALSPACE_TABLE_CACHE[xi_key] = (field_dip_1, field_dip_2, r_table)

    # --- grid / stencil precomputations (exactly as MATLAB) ---
    kcut = 2.0 * xi * np.sqrt(-np.log(errortol))
    Ngrid = np.ceil(1.0 + box * kcut / np.pi).astype(int)
    h = box / Ngrid
    P = int(np.ceil(-2.0 * np.log(errortol) / np.pi))
    offset, offsetxyz = pre_calculations(P, h)

    H0 = np.asarray(H0, dtype=np.float64).reshape(3)
    Hhat = H0 / np.linalg.norm(H0)

    C = np.zeros((Nframes,), dtype=np.complex128)
    p = np.zeros((N, 3, Nframes), dtype=np.complex128)

    if verbose >= 1:
        print(f"RunCapacitance: N={N} frames={Nframes} xi={xi} errortol={errortol:g} slab_factor={slab_factor:g} box={box.tolist()}", flush=True)

    for f in range(Nframes):
        # Initial GMRES guess aligned with Hhat (MATLAB: beta=(eps_p-1)/(eps_p+2); beta(isinf)=1)
        beta = (eps_p - 1.0) / (eps_p + 2.0)
        beta[np.isinf(eps_p)] = 1.0

        amp = 4.0 * np.pi * beta / (1.0 - beta * eta_vol)  # N×1
        Sguess = amp.reshape((-1, 1)) * Hhat.reshape((1, 3))

        if verbose >= 2:
            print(f"  frame {f+1}/{Nframes}", flush=True)

        C[f], p[:, :, f] = capacitance(
            x[:, :, f], eps_p, box, Sguess, xi, errortol,
            field_dip_1, field_dip_2, r_table,
            offset, offsetxyz, Hhat,
            slab_factor=slab_factor,
            use_scipy_gmres=use_scipy_gmres,
            verbose=verbose,
        )

    return C, p
