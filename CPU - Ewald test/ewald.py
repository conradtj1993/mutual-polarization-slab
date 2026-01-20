
"""
ewald.py

Option A consolidation: all Spectral-Ewald / real-space kernels used by the
self-consistent mutual polarization (dipolar spheres) workflow.

This module is a direct, line-faithful translation of the following MATLAB files:
- MagneticField.m
- Spread.m
- Contract.m
- Scale.m
- PreCalculations.m
- RealSpace.m
- RealSpaceTable.m
- CellList.m
- NeighborList.m

The goal is algorithmic equivalence (same equations, same control flow).
Small floating-point differences vs. MATLAB are expected because FFT and BLAS
implementations differ across environments.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Helpers: special functions (erfc, Bessel J_{3/2})
# -----------------------------------------------------------------------------
try:
    # SciPy gives fast vectorized erfc if available.
    from scipy.special import erfc as _erfc  # type: ignore
except Exception:  # pragma: no cover
    _erfc = None  # type: ignore


def erfc(x: np.ndarray) -> np.ndarray:
    """Vectorized complementary error function with a SciPy fallback."""
    x = np.asarray(x)
    if _erfc is not None:
        return _erfc(x)
    # Fallback: vectorize math.erfc (slower, but dependency-free)
    import math
    vec = np.vectorize(math.erfc, otypes=[float])
    return vec(x)


def besselj_3_over_2(x: np.ndarray) -> np.ndarray:
    """
    Bessel J_{3/2}(x), matching MATLAB besselj(1.5, x).

    Closed form:
        J_{3/2}(x) = sqrt(2/(pi x)) * (sin x / x - cos x)

    We define J_{3/2}(0) = 0 (limit).
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    mask = x != 0
    xm = x[mask]
    out[mask] = np.sqrt(2.0 / (np.pi * xm)) * (np.sin(xm) / xm - np.cos(xm))
    return out


# -----------------------------------------------------------------------------
# MATLAB-compat rounding (MATLAB round ties away from zero).
# -----------------------------------------------------------------------------
def matlab_round(x: np.ndarray) -> np.ndarray:
    """
    MATLAB-style round: round to nearest integer, ties away from zero.

    NumPy's np.round uses bankers rounding, which is NOT MATLAB-compatible.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


# -----------------------------------------------------------------------------
# Cell list and neighbor list (CellList.m, NeighborList.m)
# -----------------------------------------------------------------------------
def cell_list(x: np.ndarray, box: np.ndarray, rmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin particles into cells. Direct translation of CellList.m, but using 0-based
    cell indices for simpler Python indexing.

    Parameters
    ----------
    x : (N,3) array
        Particle positions.
    box : (3,) array
        Periodic box lengths.
    rmax : float
        Maximum distance over which to look for a pair.

    Returns
    -------
    cell : (N,3) int array
        0-based cell indices for each particle.
    Ncell : (3,) int array
        Number of cells in each dimension (>= 3).
    """
    x = np.mod(x, box)  # MATLAB: x = mod(x,box)
    Ncell = np.floor(box / rmax).astype(int)
    Ncell[Ncell < 3] = 3  # MATLAB: Ncell(Ncell < 3) = 3;
    # MATLAB: cell = floor(Ncell.*x./box) + 1; (1-based)
    # Python (0-based): floor(Ncell*x/box)
    cell = np.floor((x / box) * Ncell).astype(int)
    return cell, Ncell


def neighbor_list(cell: np.ndarray, Ncell: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a list of particle pairs in neighboring cells. Direct translation of
    NeighborList.m, returning 0-based particle indices.

    Parameters
    ----------
    cell : (N,3) int array
        0-based cell indices for each particle.
    Ncell : (3,) int array
        Number of cells in each dimension.

    Returns
    -------
    p1, p2 : (M,) int arrays
        Directed neighbor pair lists (p1[k] influenced by p2[k]).
        Self-pairs are removed (p1 != p2).
    """
    N = cell.shape[0]
    pid = np.arange(N, dtype=int)  # MATLAB: id=(1:N)' but 0-based here.

    # MATLAB: index = cell(:,1) + Ncell(1)*(cell(:,2)-1) + Nx*Ny*(cell(:,3)-1)
    # With 0-based cells, the (-1) disappears:
    Nx, Ny, Nz = (int(Ncell[0]), int(Ncell[1]), int(Ncell[2]))
    index = cell[:, 0] + Nx * cell[:, 1] + Nx * Ny * cell[:, 2]

    # MATLAB: A = sortrows([index,id,cell]); then unpack.
    # Sorting by (index, pid) is sufficient for stable cell grouping.
    order = np.lexsort((pid, index))  # primary key = index, secondary = pid
    index = index[order]
    pid = pid[order]
    cell = cell[order]

    M = Nx * Ny * Nz
    counts = np.bincount(index, minlength=M)  # MATLAB: counts = histc(index,1:prod(Ncell))
    # MATLAB:
    # cellstart = cumsum(counts);
    # cellstart = cellstart - counts + 1;   (1-based)
    # Python 0-based starts:
    cellstart = np.cumsum(counts) - counts  # length M

    p1_list: list[np.ndarray] = []
    p2_list: list[np.ndarray] = []

    # Loop over all cells
    for i in range(M):
        if counts[i] == 0:
            continue

        start_i = int(cellstart[i])
        cnt_i = int(counts[i])
        pa = pid[start_i:start_i + cnt_i]

        # cell coordinate of this cell (taken from the first particle in the cell)
        cell_coord = cell[start_i]  # (3,)

        # Search the 27 neighboring cells
        for dm in (-1, 0, 1):
            for dn in (-1, 0, 1):
                for do in (-1, 0, 1):
                    newcell = (cell_coord + np.array([dm, dn, do], dtype=int)) % Ncell
                    newidx = int(newcell[0] + Nx * newcell[1] + Nx * Ny * newcell[2])

                    if counts[newidx] == 0:
                        continue
                    start_b = int(cellstart[newidx])
                    cnt_b = int(counts[newidx])
                    pb = pid[start_b:start_b + cnt_b]

                    # MATLAB meshgrid(pa,pb) and reshape column-major produces:
                    # p1 = repeat(pa, len(pb)); p2 = tile(pb, len(pa))
                    px = np.repeat(pa, pb.size)
                    py = np.tile(pb, pa.size)

                    p1_list.append(px)
                    p2_list.append(py)

    if not p1_list:
        return np.array([], dtype=int), np.array([], dtype=int)

    p1 = np.concatenate(p1_list)
    p2 = np.concatenate(p2_list)

    # Remove self-pairs
    mask = p1 != p2
    return p1[mask], p2[mask]


# -----------------------------------------------------------------------------
# PreCalculations.m
# -----------------------------------------------------------------------------
def pre_calculations(P: int, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the indices and coordinates of the P^3 stencil nodes surrounding a node.

    Returns
    -------
    offset : (P^3, 3) int array
        Integer offsets.
    offsetxyz : (P^3, 3) float array
        Physical displacements offset * h.
    """
    linnodes = np.arange(P ** 3, dtype=int)
    half = int(np.floor((P - 1) / 2))
    offsetx = (linnodes % P) - half
    offsety = ((linnodes // P) % P) - half
    offsetz = (linnodes // (P ** 2)) - half
    offset = np.stack((offsetx, offsety, offsetz), axis=1).astype(int)
    offsetxyz = offset.astype(np.float64) * h.reshape((1, 3))
    return offset, offsetxyz


# -----------------------------------------------------------------------------
# Spread.m and Contract.m
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Fast grid precomputations (Python optimization, mathematically identical)
# -----------------------------------------------------------------------------
def prepare_grid_kernel(
    x: np.ndarray,
    Ngrid: np.ndarray,
    h: np.ndarray,
    xi: float,
    eta: np.ndarray,
    P: int,
    offset: np.ndarray,
    offsetxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute all geometry-dependent data needed by Spread/Contract for a fixed
    configuration x and fixed grid parameters (Ngrid, h, P, xi, eta).

    This is an *optimization only*: it produces exactly the same weights and node
    indices as MATLAB Spread.m / Contract.m, but computes them once per solve
    instead of once per GMRES matvec.

    Returns
    -------
    linnode : (N, P^3) int array
        0-based linear node indices in MATLAB/Fortran order:
            linnode = ix + Nx*iy + Nx*Ny*iz
    w_spread : (N, P^3) float array
        Spreading weights (same coefficients as Spread.m).
    w_contract : (N, P^3) float array
        Contracting weights (same as Spread weights multiplied by prod(h),
        matching Contract.m's trapezoidal factor).
    """
    x = np.asarray(x, dtype=np.float64)
    Ngrid = np.asarray(Ngrid, dtype=int).reshape(3)
    h = np.asarray(h, dtype=np.float64).reshape(3)
    eta = np.asarray(eta, dtype=np.float64).reshape(3)
    offset = np.asarray(offset, dtype=int)
    offsetxyz = np.asarray(offsetxyz, dtype=np.float64)

    if offset.shape[0] != P ** 3 or offsetxyz.shape[0] != P ** 3:
        raise ValueError("offset/offsetxyz shapes do not match P^3. Recompute PreCalculations.")

    Nx, Ny, Nz = int(Ngrid[0]), int(Ngrid[1]), int(Ngrid[2])
    P3 = int(P ** 3)

    # Express particle positions in grid units and find the nearest node (MATLAB round)
    xscaled = x / h.reshape((1, 3))
    node0 = matlab_round(xscaled).astype(int)  # (N,3) integer *coordinates* (not 1-based indices)

    # Node indices accounting for periodicity (MATLAB: mod(node0 + offset - 1, Ngrid) + 1)
    node = (node0[:, None, :] + offset[None, :, :] - 1) % Ngrid.reshape((1, 1, 3))  # (N,P3,3)
    linnode = (node[..., 0] + Nx * node[..., 1] + (Nx * Ny) * node[..., 2]).astype(np.int64)

    # Spreading / contracting weights (same Gaussian as MATLAB)
    const = (2.0 * xi ** 2 / np.pi) ** (3.0 / 2.0) * np.sqrt(1.0 / np.prod(eta))
    inv_eta = (1.0 / eta).reshape((1, 1, 3))

    # r = nodesxyz - x  with nodesxyz = node0*h + offsetxyz
    # Compute without explicitly forming nodesxyz:
    base = (node0.astype(np.float64) * h.reshape((1, 3)) - x)  # (N,3)
    r = base[:, None, :] + offsetxyz.reshape((1, P3, 3))       # (N,P3,3)

    exp_arg = -2.0 * xi ** 2 * np.sum((r ** 2) * inv_eta, axis=2)  # (N,P3)
    w_spread = const * np.exp(exp_arg)                               # (N,P3) real
    w_contract = w_spread * float(np.prod(h))

    return linnode, w_spread, w_contract


def spread_fast(
    m: np.ndarray,
    Ngrid: np.ndarray,
    linnode: np.ndarray,
    w_spread: np.ndarray,
) -> np.ndarray:
    """
    Vectorized equivalent of Spread.m using precomputed (linnode, w_spread).

    Returns
    -------
    H : (Nx,Ny,Nz,3) complex array
        Spread grid values for each component.
    """
    m = np.asarray(m, dtype=np.complex128)
    Ngrid = np.asarray(Ngrid, dtype=int).reshape(3)
    Nx, Ny, Nz = int(Ngrid[0]), int(Ngrid[1]), int(Ngrid[2])
    prodN = Nx * Ny * Nz

    linnode_flat = np.asarray(linnode, dtype=np.int64).reshape(-1)
    w = np.asarray(w_spread, dtype=np.float64)

    H = np.zeros((Nx, Ny, Nz, 3), dtype=np.complex128)

    # For each component, do two real bincounts (real + imag) because bincount
    # does not accept complex weights.
    for comp in range(3):
        weights = (w * m[:, comp].reshape((-1, 1))).reshape(-1)
        re = np.bincount(linnode_flat, weights=np.asarray(weights.real, dtype=np.float64), minlength=prodN)
        im = np.bincount(linnode_flat, weights=np.asarray(weights.imag, dtype=np.float64), minlength=prodN)
        H[..., comp] = (re + 1j * im).reshape((Nx, Ny, Nz), order="F")

    return H


def contract_fast(
    Htilde: np.ndarray,
    Ngrid: np.ndarray,
    linnode: np.ndarray,
    w_contract: np.ndarray,
) -> np.ndarray:
    """
    Vectorized equivalent of Contract.m using precomputed (linnode, w_contract).

    Parameters
    ----------
    Htilde : (Nx,Ny,Nz,3) complex array
        Real-space grid after inverse FFT.
    linnode : (N,P^3) int array
    w_contract : (N,P^3) float array (includes prod(h))

    Returns
    -------
    Hk : (N,3) complex array
        Reciprocal-space contribution at particle locations.
    """
    Htilde = np.asarray(Htilde, dtype=np.complex128)
    w = np.asarray(w_contract, dtype=np.float64)
    linnode = np.asarray(linnode, dtype=np.int64)

    N = linnode.shape[0]
    Hk = np.zeros((N, 3), dtype=np.complex128)

    # Flatten each component in MATLAB/Fortran order and gather via linnode
    for comp in range(3):
        flat = Htilde[..., comp].ravel(order="F")
        vals = flat[linnode]  # (N,P^3)
        Hk[:, comp] = np.sum(w * vals, axis=1)

    return Hk

def spread(
    x: np.ndarray,
    m: np.ndarray,
    Ngrid: np.ndarray,
    h: np.ndarray,
    xi: float,
    eta: np.ndarray,
    P: int,
    offset: np.ndarray,
    offsetxyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Spread particle dipoles to a regular grid (Spectral Ewald spreading).

    Direct translation of Spread.m. Uses the same "nearestnode = round(x/h)" convention
    and the same periodic wrapping formula for node indices.
    """
    N = x.shape[0]
    Nx, Ny, Nz = (int(Ngrid[0]), int(Ngrid[1]), int(Ngrid[2]))

    Hx = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Hy = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
    Hz = np.zeros((Nx, Ny, Nz), dtype=np.complex128)

    xscaled = x / h.reshape((1, 3))
    nearestnode = matlab_round(xscaled).astype(int)

    const = (2.0 * xi ** 2 / np.pi) ** (3.0 / 2.0) * np.sqrt(1.0 / np.prod(eta))

    inv_eta = 1.0 / eta.reshape((1, 3))

    for n in range(N):
        node0 = nearestnode[n]  # may be 0..Ngrid
        nodesxyz = (node0.astype(np.float64) * h).reshape((1, 3)) + offsetxyz  # (P^3,3)
        r = nodesxyz - x[n].reshape((1, 3))
        # MATLAB: exp(-2*xi^2*r.^2*(1./eta'))
        exp_arg = -2.0 * xi ** 2 * np.sum((r ** 2) * inv_eta, axis=1)
        Hcoeff = const * np.exp(exp_arg)  # (P^3,)

        # MATLAB node index (1-based): node = mod(node0 + offset - 1, Ngrid) + 1
        # Python 0-based indices into arrays:
        node = (node0.reshape((1, 3)) + offset - 1) % Ngrid.reshape((1, 3))
        ix, iy, iz = node[:, 0], node[:, 1], node[:, 2]

        np.add.at(Hx, (ix, iy, iz), Hcoeff * m[n, 0])
        np.add.at(Hy, (ix, iy, iz), Hcoeff * m[n, 1])
        np.add.at(Hz, (ix, iy, iz), Hcoeff * m[n, 2])

    return Hx, Hy, Hz


def contract(
    x: np.ndarray,
    Ngrid: np.ndarray,
    h: np.ndarray,
    xi: float,
    eta: np.ndarray,
    P: int,
    Htilde: np.ndarray,
    offset: np.ndarray,
    offsetxyz: np.ndarray,
) -> np.ndarray:
    """
    Contract the gridded values to the field at particle centers.

    Direct translation of Contract.m (including the trapezoidal factor prod(h)).
    """
    N = x.shape[0]
    Hk = np.zeros((N, 3), dtype=np.complex128)

    xscaled = x / h.reshape((1, 3))
    nearestnode = matlab_round(xscaled).astype(int)

    Hxtilde = Htilde[..., 0]
    Hytilde = Htilde[..., 1]
    Hztilde = Htilde[..., 2]

    const = (2.0 * xi ** 2 / np.pi) ** (3.0 / 2.0) * np.sqrt(1.0 / np.prod(eta))
    inv_eta = 1.0 / eta.reshape((1, 3))

    for n in range(N):
        node0 = nearestnode[n]
        nodesxyz = (node0.astype(np.float64) * h).reshape((1, 3)) + offsetxyz
        r = nodesxyz - x[n].reshape((1, 3))
        exp_arg = -2.0 * xi ** 2 * np.sum((r ** 2) * inv_eta, axis=1)
        Hcoeff = const * np.exp(exp_arg)

        # Trapezoidal rule
        Hcoeff = Hcoeff * float(np.prod(h))

        node = (node0.reshape((1, 3)) + offset - 1) % Ngrid.reshape((1, 3))
        ix, iy, iz = node[:, 0], node[:, 1], node[:, 2]

        Hk[n, 0] = np.sum(Hcoeff * Hxtilde[ix, iy, iz])
        Hk[n, 1] = np.sum(Hcoeff * Hytilde[ix, iy, iz])
        Hk[n, 2] = np.sum(Hcoeff * Hztilde[ix, iy, iz])

    return Hk


# -----------------------------------------------------------------------------
# Scale.m
# -----------------------------------------------------------------------------
def scale(fH: np.ndarray, k: np.ndarray, Ngrid: np.ndarray, xi: float, eta: np.ndarray) -> np.ndarray:
    """
    Scale the gridded values for the field (reciprocal space kernel).

    Direct translation of Scale.m.
    """
    k2 = np.sum(k ** 2, axis=3)  # (Nx,Ny,Nz)
    Nx, Ny, Nz = (int(Ngrid[0]), int(Ngrid[1]), int(Ngrid[2]))

    # MATLAB:
    # k0x = ceil((Ngrid(1) - 1)/2) + 1; (1-based)
    # In 0-based Python:
    k0x = int(np.ceil((Nx - 1) / 2.0))
    k0y = int(np.ceil((Ny - 1) / 2.0))
    k0z = int(np.ceil((Nz - 1) / 2.0))

    # khat = k./repmat(sqrt(k2),...,3)
    sqrt_k2 = np.sqrt(k2)
    denom = sqrt_k2.copy()
    denom[denom == 0] = 1.0
    khat = (k / denom[..., np.newaxis]).astype(np.complex128)
    khat[k0x, k0y, k0z, :] = 0.0

    # etak2 = dot(repmat(reshape((1-eta),1,1,1,3),...),k.^2,4);
    etak2 = (1.0 - eta[0]) * (k[..., 0] ** 2) + (1.0 - eta[1]) * (k[..., 1] ** 2) + (1.0 - eta[2]) * (k[..., 2] ** 2)

    # Htildecoeff = 9*pi./(2*sqrt(k2)).*besselj(1+1/2,sqrt(k2)).^2.*exp(-etak2/(4*xi^2))./k2;
    # besselj(1.5, sqrt(k2)) implemented as J_{3/2}
    x = np.sqrt(k2)
    J = besselj_3_over_2(x)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Htildecoeff = (9.0 * np.pi) / (2.0 * np.sqrt(k2)) * (J ** 2) * np.exp(-etak2 / (4.0 * xi ** 2)) / k2

    Htildecoeff[k0x, k0y, k0z] = 0.0

    # fHtilde = repmat(Htildecoeff.*dot(khat,fH,4),[1,1,1,3]).*khat;
    dot_khat_fH = np.sum(khat * fH, axis=3)  # (Nx,Ny,Nz)
    fHtilde = (Htildecoeff * dot_khat_fH)[..., np.newaxis] * khat

    # Ensure k=0 mode zeroed (already), and NaNs suppressed
    fHtilde[k0x, k0y, k0z, :] = 0.0
    fHtilde = np.nan_to_num(fHtilde, copy=False)

    return fHtilde



# -----------------------------------------------------------------------------
# Fast reciprocal-space precomputations (Python optimization, mathematically identical)
# -----------------------------------------------------------------------------
def precompute_kspace(
    Ngrid: np.ndarray,
    box: np.ndarray,
    xi: float,
    eta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute the reciprocal-space kernel data used in Scale.m.

    This avoids rebuilding the k-grid, khat, and the scalar scaling coefficient on
    every MagneticField call during GMRES. It is algebraically identical to the
    MATLAB code in MagneticField.m + Scale.m.

    Returns
    -------
    khat : (Nx,Ny,Nz,3) complex array
        Unit wave-vector directions (k/|k|), with the k=0 entry set to 0.
    Htildecoeff : (Nx,Ny,Nz) float array
        Scalar coefficient multiplying dot(khat,fH) in Scale.m, with k=0 set to 0.
    """
    Ngrid = np.asarray(Ngrid, dtype=int).reshape(3)
    box = np.asarray(box, dtype=np.float64).reshape(3)
    eta = np.asarray(eta, dtype=np.float64).reshape(3)

    Nx, Ny, Nz = int(Ngrid[0]), int(Ngrid[1]), int(Ngrid[2])

    # MATLAB k-vectors, matched to fftshift ordering
    kx = np.arange(-int(np.ceil((Nx - 1) / 2.0)), int(np.floor((Nx - 1) / 2.0)) + 1) * (2.0 * np.pi / box[0])
    ky = np.arange(-int(np.ceil((Ny - 1) / 2.0)), int(np.floor((Ny - 1) / 2.0)) + 1) * (2.0 * np.pi / box[1])
    kz = np.arange(-int(np.ceil((Nz - 1) / 2.0)), int(np.floor((Nz - 1) / 2.0)) + 1) * (2.0 * np.pi / box[2])
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k = np.stack((KX, KY, KZ), axis=3)  # (Nx,Ny,Nz,3)

    k2 = np.sum(k ** 2, axis=3)  # (Nx,Ny,Nz)

    # Index of the k=0 entry in fftshift ordering
    k0x = int(np.ceil((Nx - 1) / 2.0))
    k0y = int(np.ceil((Ny - 1) / 2.0))
    k0z = int(np.ceil((Nz - 1) / 2.0))

    sqrt_k2 = np.sqrt(k2)
    denom = sqrt_k2.copy()
    denom[denom == 0] = 1.0

    khat = (k / denom[..., np.newaxis]).astype(np.complex128)
    khat[k0x, k0y, k0z, :] = 0.0

    # etak2 = dot((1-eta), k.^2) along the 4th dimension
    etak2 = (1.0 - eta[0]) * (k[..., 0] ** 2) + (1.0 - eta[1]) * (k[..., 1] ** 2) + (1.0 - eta[2]) * (k[..., 2] ** 2)

    x = np.sqrt(k2)
    J = besselj_3_over_2(x)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Htildecoeff = (9.0 * np.pi) / (2.0 * np.sqrt(k2)) * (J ** 2) * np.exp(-etak2 / (4.0 * xi ** 2)) / k2

    Htildecoeff[k0x, k0y, k0z] = 0.0
    Htildecoeff = np.nan_to_num(Htildecoeff, copy=False)

    return khat, np.asarray(Htildecoeff, dtype=np.float64)


def scale_fast(
    fH: np.ndarray,
    khat: np.ndarray,
    Htildecoeff: np.ndarray,
) -> np.ndarray:
    """
    Fast equivalent of Scale.m given precomputed (khat, Htildecoeff).
    """
    fH = np.asarray(fH, dtype=np.complex128)
    khat = np.asarray(khat, dtype=np.complex128)
    Htildecoeff = np.asarray(Htildecoeff, dtype=np.float64)

    dot_khat_fH = np.sum(khat * fH, axis=3)
    fHtilde = (Htildecoeff * dot_khat_fH)[..., np.newaxis] * khat
    return fHtilde

# -----------------------------------------------------------------------------
# RealSpaceTable.m
# -----------------------------------------------------------------------------
def real_space_table(r: np.ndarray, xi: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Tabulate the real space contributions as a function of particle separation.

    Direct translation of RealSpaceTable.m. Input r should NOT include 0; this function
    appends self-terms as in MATLAB.
    """
    r = np.asarray(r, dtype=np.float64).reshape(-1)

    # --- Potential/Charge Coupling ---
    exppolyp = -(r + 2.0) / (32.0 * np.pi ** (3.0 / 2.0) * xi * r)
    exppolym = -(r - 2.0) / (32.0 * np.pi ** (3.0 / 2.0) * xi * r)
    exppoly0 = 1.0 / (16.0 * np.pi ** (3.0 / 2.0) * xi)

    erfpolyp = (2.0 * xi ** 2 * (r + 2.0) ** 2 + 1.0) / (64.0 * np.pi * xi ** 2 * r)
    erfpolym = (2.0 * xi ** 2 * (r - 2.0) ** 2 + 1.0) / (64.0 * np.pi * xi ** 2 * r)
    erfpoly0 = -(2.0 * xi ** 2 * r ** 2 + 1.0) / (32.0 * np.pi * xi ** 2 * r)

    regpoly = -1.0 / (4.0 * np.pi * r) + (4.0 - r) / (16.0 * np.pi)

    pot_charge = (
        exppolyp * np.exp(-(r + 2.0) ** 2 * xi ** 2)
        + exppolym * np.exp(-(r - 2.0) ** 2 * xi ** 2)
        + exppoly0 * np.exp(-r ** 2 * xi ** 2)
        + erfpolyp * erfc((r + 2.0) * xi)
        + erfpolym * erfc((r - 2.0) * xi)
        + erfpoly0 * erfc(r * xi)
        + (r < 2.0) * regpoly
    )

    # --- Potential/Dipole or Field/Charge coupling ---
    exppolyp = (1.0 / (256.0 * np.pi ** (3.0 / 2.0) * xi ** 3 * r ** 2)) * (
        -6.0 * xi ** 2 * r ** 3
        - 4.0 * xi ** 2 * r ** 2
        + (-3.0 + 8.0 * xi ** 2) * r
        + 2.0 * (1.0 - 8.0 * xi ** 2)
    )
    exppolym = (1.0 / (256.0 * np.pi ** (3.0 / 2.0) * xi ** 3 * r ** 2)) * (
        -6.0 * xi ** 2 * r ** 3
        + 4.0 * xi ** 2 * r ** 2
        + (-3.0 + 8.0 * xi ** 2) * r
        - 2.0 * (1.0 - 8.0 * xi ** 2)
    )
    exppoly0 = 3.0 * (2.0 * r ** 2 * xi ** 2 + 1.0) / (128.0 * np.pi ** (3.0 / 2.0) * xi ** 3 * r)

    erfpolyp = (1.0 / (512.0 * np.pi * xi ** 4 * r ** 2)) * (
        12.0 * xi ** 4 * r ** 4
        + 32.0 * xi ** 4 * r ** 3
        + 12.0 * xi ** 2 * r ** 2
        - 3.0
        + 64.0 * xi ** 4
    )
    erfpolym = (1.0 / (512.0 * np.pi * xi ** 4 * r ** 2)) * (
        12.0 * xi ** 4 * r ** 4
        - 32.0 * xi ** 4 * r ** 3
        + 12.0 * xi ** 2 * r ** 2
        - 3.0
        + 64.0 * xi ** 4
    )
    erfpoly0 = -3.0 * (4.0 * xi ** 4 * r ** 4 + 4.0 * xi ** 2 * r ** 2 - 1.0) / (256.0 * np.pi * xi ** 4 * r ** 2)

    regpoly = -1.0 / (4.0 * np.pi * r ** 2) + r / (8.0 * np.pi) * (1.0 - 3.0 / 8.0 * r)

    pot_dip = (
        exppolyp * np.exp(-(r + 2.0) ** 2 * xi ** 2)
        + exppolym * np.exp(-(r - 2.0) ** 2 * xi ** 2)
        + exppoly0 * np.exp(-r ** 2 * xi ** 2)
        + erfpolyp * erfc((r + 2.0) * xi)
        + erfpolym * erfc((r - 2.0) * xi)
        + erfpoly0 * erfc(r * xi)
        + (r < 2.0) * regpoly
    )

    # --- Field/Dipole coupling: I-rr component ---
    exppolyp = (1.0 / (1024.0 * np.pi ** (3.0 / 2.0) * xi ** 5 * r ** 3)) * (
        4.0 * xi ** 4 * r ** 5
        - 8.0 * xi ** 4 * r ** 4
        + 8.0 * xi ** 2 * (2.0 - 7.0 * xi ** 2) * r ** 3
        - 8.0 * xi ** 2 * (3.0 + 2.0 * xi ** 2) * r ** 2
        + (3.0 - 12.0 * xi ** 2 + 32.0 * xi ** 4) * r
        + 2.0 * (3.0 + 4.0 * xi ** 2 - 32.0 * xi ** 4)
    )
    exppolym = (1.0 / (1024.0 * np.pi ** (3.0 / 2.0) * xi ** 5 * r ** 3)) * (
        4.0 * xi ** 4 * r ** 5
        + 8.0 * xi ** 4 * r ** 4
        + 8.0 * xi ** 2 * (2.0 - 7.0 * xi ** 2) * r ** 3
        + 8.0 * xi ** 2 * (3.0 + 2.0 * xi ** 2) * r ** 2
        + (3.0 - 12.0 * xi ** 2 + 32.0 * xi ** 4) * r
        - 2.0 * (3.0 + 4.0 * xi ** 2 - 32.0 * xi ** 4)
    )
    exppoly0 = (1.0 / (512.0 * np.pi ** (3.0 / 2.0) * xi ** 5 * r ** 2)) * (
        -4.0 * xi ** 4 * r ** 4
        - 8.0 * xi ** 2 * (2.0 - 9.0 * xi ** 2) * r ** 2
        - 3.0
        + 36.0 * xi ** 2
    )

    erfpolyp = (1.0 / (2048.0 * np.pi * xi ** 6 * r ** 3)) * (
        -8.0 * xi ** 6 * r ** 6
        - 36.0 * xi ** 4 * (1.0 - 4.0 * xi ** 2) * r ** 4
        + 256.0 * xi ** 6 * r ** 3
        - 18.0 * xi ** 2 * (1.0 - 8.0 * xi ** 2) * r ** 2
        + 3.0
        - 36.0 * xi ** 2
        + 256.0 * xi ** 6
    )
    erfpolym = (1.0 / (2048.0 * np.pi * xi ** 6 * r ** 3)) * (
        -8.0 * xi ** 6 * r ** 6
        - 36.0 * xi ** 4 * (1.0 - 4.0 * xi ** 2) * r ** 4
        - 256.0 * xi ** 6 * r ** 3
        - 18.0 * xi ** 2 * (1.0 - 8.0 * xi ** 2) * r ** 2
        + 3.0
        - 36.0 * xi ** 2
        + 256.0 * xi ** 6
    )
    erfpoly0 = (1.0 / (1024.0 * np.pi * xi ** 6 * r ** 3)) * (
        8.0 * xi ** 6 * r ** 6
        + 36.0 * xi ** 4 * (1.0 - 4.0 * xi ** 2) * r ** 4
        + 18.0 * xi ** 2 * (1.0 - 8.0 * xi ** 2) * r ** 2
        - 3.0
        + 36.0 * xi ** 2
    )

    regpoly = -1.0 / (4.0 * np.pi * r ** 3) + 1.0 / (4.0 * np.pi) * (1.0 - 9.0 * r / 16.0 + r ** 3 / 32.0)

    field_dip_1 = (
        exppolyp * np.exp(-(r + 2.0) ** 2 * xi ** 2)
        + exppolym * np.exp(-(r - 2.0) ** 2 * xi ** 2)
        + exppoly0 * np.exp(-r ** 2 * xi ** 2)
        + erfpolyp * erfc((r + 2.0) * xi)
        + erfpolym * erfc((r - 2.0) * xi)
        + erfpoly0 * erfc(r * xi)
        + (r < 2.0) * regpoly
    )

    # --- Field/Dipole coupling: rr component ---
    exppolyp = (1.0 / (512.0 * np.pi ** (3.0 / 2.0) * xi ** 5 * r ** 3)) * (
        8.0 * xi ** 4 * r ** 5
        - 16.0 * xi ** 4 * r ** 4
        + 2.0 * xi ** 2 * (7.0 - 20.0 * xi ** 2) * r ** 3
        - 4.0 * xi ** 2 * (3.0 - 4.0 * xi ** 2) * r ** 2
        - (3.0 - 12.0 * xi ** 2 + 32.0 * xi ** 4) * r
        - 2.0 * (3.0 + 4.0 * xi ** 2 - 32.0 * xi ** 4)
    )
    exppolym = (1.0 / (512.0 * np.pi ** (3.0 / 2.0) * xi ** 5 * r ** 3)) * (
        8.0 * xi ** 4 * r ** 5
        + 16.0 * xi ** 4 * r ** 4
        + 2.0 * xi ** 2 * (7.0 - 20.0 * xi ** 2) * r ** 3
        + 4.0 * xi ** 2 * (3.0 - 4.0 * xi ** 2) * r ** 2
        - (3.0 - 12.0 * xi ** 2 + 32.0 * xi ** 4) * r
        + 2.0 * (3.0 + 4.0 * xi ** 2 - 32.0 * xi ** 4)
    )
    exppoly0 = (1.0 / (256.0 * np.pi ** (3.0 / 2.0) * xi ** 5 * r ** 2)) * (
        -8.0 * xi ** 4 * r ** 4
        - 2.0 * xi ** 2 * (7.0 - 36.0 * xi ** 2) * r ** 2
        + 3.0
        - 36.0 * xi ** 2
    )

    erfpolyp = (1.0 / (1024.0 * np.pi * xi ** 6 * r ** 3)) * (
        -16.0 * xi ** 6 * r ** 6
        - 36.0 * xi ** 4 * (1.0 - 4.0 * xi ** 2) * r ** 4
        + 128.0 * xi ** 6 * r ** 3
        - 3.0
        + 36.0 * xi ** 2
        - 256.0 * xi ** 6
    )
    erfpolym = (1.0 / (1024.0 * np.pi * xi ** 6 * r ** 3)) * (
        -16.0 * xi ** 6 * r ** 6
        - 36.0 * xi ** 4 * (1.0 - 4.0 * xi ** 2) * r ** 4
        - 128.0 * xi ** 6 * r ** 3
        - 3.0
        + 36.0 * xi ** 2
        - 256.0 * xi ** 6
    )
    erfpoly0 = (1.0 / (512.0 * np.pi * xi ** 6 * r ** 3)) * (
        16.0 * xi ** 6 * r ** 6
        + 36.0 * xi ** 4 * (1.0 - 4.0 * xi ** 2) * r ** 4
        + 3.0
        - 36.0 * xi ** 2
    )

    regpoly = 1.0 / (2.0 * np.pi * r ** 3) + 1.0 / (4.0 * np.pi) * (1.0 - 9.0 * r / 8.0 + r ** 3 / 8.0)

    field_dip_2 = (
        exppolyp * np.exp(-(r + 2.0) ** 2 * xi ** 2)
        + exppolym * np.exp(-(r - 2.0) ** 2 * xi ** 2)
        + exppoly0 * np.exp(-r ** 2 * xi ** 2)
        + erfpolyp * erfc((r + 2.0) * xi)
        + erfpolym * erfc((r - 2.0) * xi)
        + erfpoly0 * erfc(r * xi)
        + (r < 2.0) * regpoly
    )

    # --- Field/Dipole Force: coefficient multiplying -(mi*mj)r and other terms ---
    exppolyp = 3.0 / (1024.0 * np.pi ** (3.0 / 2.0) * xi ** 5 * r ** 4) * (
        4.0 * xi ** 4 * r ** 5
        - 8.0 * xi ** 4 * r ** 4
        + 4.0 * xi ** 2 * (1.0 - 2.0 * xi ** 2) * r ** 3
        + 16.0 * xi ** 4 * r ** 2
        - (3.0 - 12.0 * xi ** 2 + 32.0 * xi ** 4) * r
        - 2.0 * (3.0 + 4.0 * xi ** 2 - 32.0 * xi ** 4)
    )
    exppolym = 3.0 / (1024.0 * np.pi ** (3.0 / 2.0) * xi ** 5 * r ** 4) * (
        4.0 * xi ** 4 * r ** 5
        + 8.0 * xi ** 4 * r ** 4
        + 4.0 * xi ** 2 * (1.0 - 2.0 * xi ** 2) * r ** 3
        - 16.0 * xi ** 4 * r ** 2
        - (3.0 - 12.0 * xi ** 2 + 32.0 * xi ** 4) * r
        + 2.0 * (3.0 + 4.0 * xi ** 2 - 32.0 * xi ** 4)
    )
    exppoly0 = 3.0 / (512.0 * np.pi ** (3.0 / 2.0) * xi ** 5 * r ** 3) * (
        -4.0 * xi ** 4 * r ** 4
        - 4.0 * xi ** 2 * (1.0 - 6.0 * xi ** 2) * r ** 2
        + 3.0
        - 36.0 * xi ** 2
    )

    erfpolyp = 3.0 / (2048.0 * np.pi * xi ** 6 * r ** 4) * (
        -8.0 * xi ** 6 * r ** 6
        - 12.0 * xi ** 4 * (1.0 - 4.0 * xi ** 2) * r ** 4
        + 6.0 * xi ** 2 * (1.0 - 8.0 * xi ** 2) * r ** 2
        - 3.0
        + 36.0 * xi ** 2
        - 256.0 * xi ** 6
    )
    erfpolym = 3.0 / (2048.0 * np.pi * xi ** 6 * r ** 4) * (
        -8.0 * xi ** 6 * r ** 6
        - 12.0 * xi ** 4 * (1.0 - 4.0 * xi ** 2) * r ** 4
        + 6.0 * xi ** 2 * (1.0 - 8.0 * xi ** 2) * r ** 2
        - 3.0
        + 36.0 * xi ** 2
        - 256.0 * xi ** 6
    )
    erfpoly0 = 3.0 / (1024.0 * np.pi * xi ** 6 * r ** 4) * (
        8.0 * xi ** 6 * r ** 6
        + 12.0 * xi ** 4 * (1.0 - 4.0 * xi ** 2) * r ** 4
        - 6.0 * xi ** 2 * (1.0 - 8.0 * xi ** 2) * r ** 2
        + 3.0
        - 36.0 * xi ** 2
    )

    regpoly = 3.0 / (4.0 * np.pi * r ** 4) - 3.0 / (64.0 * np.pi) * (3.0 - r ** 2 / 2.0)

    field_dip_force_1 = (
        exppolyp * np.exp(-(r + 2.0) ** 2 * xi ** 2)
        + exppolym * np.exp(-(r - 2.0) ** 2 * xi ** 2)
        + exppoly0 * np.exp(-r ** 2 * xi ** 2)
        + erfpolyp * erfc((r + 2.0) * xi)
        + erfpolym * erfc((r - 2.0) * xi)
        + erfpoly0 * erfc(r * xi)
        + (r < 2.0) * regpoly
    )

    # --- Field/Dipole Force from: coefficient multiplying -(mi*r)(mj*r)r ---
    exppolyp = 9.0 / (1024.0 * np.pi ** (3.0 / 2.0) * xi ** 5 * r ** 4) * (
        4.0 * xi ** 4 * r ** 5
        - 8.0 * xi ** 4 * r ** 4
        + 8.0 * xi ** 4 * r ** 3
        + 8.0 * xi ** 2 * (1.0 - 2.0 * xi ** 2) * r ** 2
        + (3.0 - 12.0 * xi ** 2 + 32.0 * xi ** 4) * r
        + 2.0 * (3.0 + 4.0 * xi ** 2 - 32.0 * xi ** 4)
    )
    exppolym = 9.0 / (1024.0 * np.pi ** (3.0 / 2.0) * xi ** 5 * r ** 4) * (
        4.0 * xi ** 4 * r ** 5
        + 8.0 * xi ** 4 * r ** 4
        + 8.0 * xi ** 4 * r ** 3
        - 8.0 * xi ** 2 * (1.0 - 2.0 * xi ** 2) * r ** 2
        + (3.0 - 12.0 * xi ** 2 + 32.0 * xi ** 4) * r
        - 2.0 * (3.0 + 4.0 * xi ** 2 - 32.0 * xi ** 4)
    )
    exppoly0 = 9.0 / (512.0 * np.pi ** (3.0 / 2.0) * xi ** 5 * r ** 3) * (
        -4.0 * xi ** 4 * r ** 4
        + 8.0 * xi ** 4 * r ** 2
        - 3.0
        + 36.0 * xi ** 2
    )

    erfpolyp = 9.0 / (2048.0 * np.pi * xi ** 6 * r ** 4) * (
        -8.0 * xi ** 6 * r ** 6
        - 4.0 * xi ** 4 * (1.0 - 4.0 * xi ** 2) * r ** 4
        - 2.0 * xi ** 2 * (1.0 - 8.0 * xi ** 2) * r ** 2
        + 3.0
        - 36.0 * xi ** 2
        + 256.0 * xi ** 6
    )
    erfpolym = 9.0 / (2048.0 * np.pi * xi ** 6 * r ** 4) * (
        -8.0 * xi ** 6 * r ** 6
        - 4.0 * xi ** 4 * (1.0 - 4.0 * xi ** 2) * r ** 4
        - 2.0 * xi ** 2 * (1.0 - 8.0 * xi ** 2) * r ** 2
        + 3.0
        - 36.0 * xi ** 2
        + 256.0 * xi ** 6
    )
    erfpoly0 = 9.0 / (1024.0 * np.pi * xi ** 6 * r ** 4) * (
        8.0 * xi ** 6 * r ** 6
        + 4.0 * xi ** 4 * (1.0 - 4.0 * xi ** 2) * r ** 4
        + 2.0 * xi ** 2 * (1.0 - 8.0 * xi ** 2) * r ** 2
        - 3.0
        + 36.0 * xi ** 2
    )

    regpoly = -9.0 / (4.0 * np.pi * r ** 4) - 9.0 / (64.0 * np.pi) * (1.0 - r ** 2 / 2.0)

    field_dip_force_2 = (
        exppolyp * np.exp(-(r + 2.0) ** 2 * xi ** 2)
        + exppolym * np.exp(-(r - 2.0) ** 2 * xi ** 2)
        + exppoly0 * np.exp(-r ** 2 * xi ** 2)
        + erfpolyp * erfc((r + 2.0) * xi)
        + erfpolym * erfc((r - 2.0) * xi)
        + erfpoly0 * erfc(r * xi)
        + (r < 2.0) * regpoly
    )

    # --- Self terms ---
    self_pc = (1.0 - np.exp(-4.0 * xi ** 2)) / (8.0 * np.pi ** (3.0 / 2.0) * xi) + erfc(2.0 * xi) / (4.0 * np.pi)
    pot_charge = np.concatenate(([self_pc], pot_charge))

    pot_dip = np.concatenate(([0.0], pot_dip))

    self_fd = (-1.0 + 6.0 * xi ** 2 + (1.0 - 2.0 * xi ** 2) * np.exp(-4.0 * xi ** 2)) / (16.0 * np.pi ** (3.0 / 2.0) * xi ** 3) + erfc(2.0 * xi) / (4.0 * np.pi)
    field_dip_1 = np.concatenate(([self_fd], field_dip_1))
    field_dip_2 = np.concatenate(([self_fd], field_dip_2))

    field_dip_force_1 = np.concatenate(([0.0], field_dip_force_1))
    field_dip_force_2 = np.concatenate(([0.0], field_dip_force_2))

    return pot_charge, pot_dip, field_dip_1, field_dip_2, field_dip_force_1, field_dip_force_2




# -----------------------------------------------------------------------------
# Fast real-space precomputation (geometry-only) + fast evaluation
# -----------------------------------------------------------------------------
def prepare_real_space_pairs(
    x: np.ndarray,
    box: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    rc: float,
    perp: np.ndarray,
    para: np.ndarray,
    rvals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute geometry-dependent quantities for the real-space contribution once,
    so that GMRES matvecs can evaluate RealSpace without looping over candidate
    pairs in Python.

    This is mathematically identical to RealSpace.m because it keeps *exactly*
    the same cutoff rule (d < rc) and the same linear interpolation in the
    tabulated coefficients. The only change is moving the work that depends
    only on (x, box) outside the iterative solve.

    Returns
    -------
    p1f, p2f : (M,) int arrays
        Filtered directed pair indices with d < rc.
    rhat : (M,3) float array
        Unit separation vector from p2 -> p1 (minimum-image).
    A, B : (M,) float arrays
        Interpolated table coefficients (perp/para) for each separation d.
    """
    x = np.asarray(x, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=int).reshape(-1)
    p2 = np.asarray(p2, dtype=int).reshape(-1)

    if p1.size == 0:
        empty_i = np.array([], dtype=int)
        empty_f = np.array([], dtype=np.float64)
        return empty_i, empty_i, empty_f.reshape((0, 3)), empty_f, empty_f

    # Minimum-image displacement r = x[p1]-x[p2] - box*fix(2*r/box)
    r = x[p1] - x[p2]
    r = r - box.reshape((1, 3)) * np.trunc(2.0 * r / box.reshape((1, 3)))

    d = np.sqrt(np.sum(r * r, axis=1))
    mask = d < float(rc)

    p1f = p1[mask]
    p2f = p2[mask]
    r = r[mask]
    d = d[mask]

    if d.size == 0:
        empty_f = np.array([], dtype=np.float64)
        return p1f, p2f, empty_f.reshape((0, 3)), empty_f, empty_f

    rhat = r / d.reshape((-1, 1))

    # Interpolation indices: first index where rvals >= d (MATLAB: find(floor(rvals/d),1))
    interpind = np.searchsorted(rvals, d, side="left")

    if np.any(interpind <= 0) or np.any(interpind >= rvals.size):
        dmin = float(np.min(d)) if d.size else float("nan")
        dmax = float(np.max(d)) if d.size else float("nan")
        raise ValueError(
            "RealSpace interpolation: pair separations fall outside rvals table. "
            f"d range = [{dmin:.6g}, {dmax:.6g}], rvals range = "
            f"[{float(rvals[0]):.6g}, {float(rvals[-1]):.6g}]. "
            "Increase r_table max (in RunCapacitance) or adjust xi/errortol."
        )

    rlow = rvals[interpind - 1]
    rhigh = rvals[interpind]
    t = (d - rlow) / (rhigh - rlow)

    A = perp[interpind - 1] * (1.0 - t) + perp[interpind] * t
    B = para[interpind - 1] * (1.0 - t) + para[interpind] * t

    return p1f, p2f, rhat, A, B


def real_space_fast(
    m: np.ndarray,
    lambdap: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    rhat: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    self_perp: float,
) -> np.ndarray:
    """
    Fast evaluation of the real-space contribution using precomputed geometry
    from prepare_real_space_pairs().

    This computes exactly the same quantity as RealSpace.m:

        Hr = -3/(4*pi*(1-lambdap))*m  +  m*self_perp
        Hr[p1] += A*(mj - (mj·rhat) rhat) + B*(mj·rhat) rhat

    where mj = m[p2].

    Parameters
    ----------
    m : (N,3) array (complex)
        Current dipoles in the GMRES matvec.
    lambdap : (N,) array
        Particle permittivities.
    p1, p2, rhat, A, B : arrays
        Precomputed pair geometry and coefficients.
    self_perp : float
        The self-term coefficient, i.e. perp(1) in MATLAB (perp[0] in Python).

    Returns
    -------
    Hr : (N,3) complex array
    """
    m = np.asarray(m, dtype=np.complex128)
    lambdap = np.asarray(lambdap, dtype=np.float64).reshape(-1)

    Hr = (-3.0 / (4.0 * np.pi * (1.0 - lambdap))).reshape((-1, 1)) * m
    Hr = Hr + m * float(self_perp)

    if p1.size == 0:
        return Hr

    mj = m[p2]
    mj_dot_r = np.sum(mj * rhat, axis=1)  # (M,)
    BA = (B - A) * mj_dot_r               # (M,)

    contrib = A.reshape((-1, 1)) * mj + BA.reshape((-1, 1)) * rhat
    # Scatter-add into Hr at indices p1
    np.add.at(Hr, p1, contrib)

    return Hr
# -----------------------------------------------------------------------------
# RealSpace.m
# -----------------------------------------------------------------------------
def real_space(
    x: np.ndarray,
    m: np.ndarray,
    lambdap: np.ndarray,
    box: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    rc: float,
    perp: np.ndarray,
    para: np.ndarray,
    rvals: np.ndarray,
    real_precomp: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """
    Real space contribution to the field.

    Direct translation of RealSpace.m.
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(m, dtype=np.complex128)
    lambdap = np.asarray(lambdap, dtype=np.float64).reshape(-1)
    box = np.asarray(box, dtype=np.float64).reshape(3)

    Hr = (-3.0 / (4.0 * np.pi * (1.0 - lambdap))).reshape((-1, 1)) * m  # dielectric contribution
    Hr = Hr + m * perp[0]  # self term (perp(1) in MATLAB)

    # Loop over neighbor pairs
    for i in range(p1.size):
        i1 = int(p1[i])
        i2 = int(p2[i])

        r = x[i1] - x[i2]
        # MATLAB: r = r - box.*fix(2*r./box);
        r = r - box * np.trunc(2.0 * r / box)

        d = float(np.sqrt(np.dot(r, r)))
        if d == 0.0:
            continue
        rhat = r / d

        if d < rc:
            mj = m[i2]

            # MATLAB: interpind = find(floor(rvals/d),1);
            # Equivalent: first index where rvals >= d
            interpind = int(np.searchsorted(rvals, d, side="left"))
            if interpind <= 0:
                continue
            if interpind >= rvals.size:
                raise ValueError(
                    "RealSpace interpolation: pair separation d exceeds the maximum tabulated rvals. "
                    "Increase r_table max (in RunCapacitance) or reduce rc/tolerance."
                )
            rhigh = float(rvals[interpind])
            rlow = float(rvals[interpind - 1])

            # Linear interpolation
            A = perp[interpind - 1] * (rhigh - d) / (rhigh - rlow) + perp[interpind] * (d - rlow) / (rhigh - rlow)
            B = para[interpind - 1] * (rhigh - d) / (rhigh - rlow) + para[interpind] * (d - rlow) / (rhigh - rlow)

            mj_dot_r = np.dot(mj, rhat)
            Hr[i1] = Hr[i1] + A * (mj - mj_dot_r * rhat) + B * (mj_dot_r * rhat)

    return Hr


# -----------------------------------------------------------------------------
# MagneticField.m
# -----------------------------------------------------------------------------

def magnetic_field(
    x: np.ndarray,
    m: np.ndarray,
    lambdap: np.ndarray,
    box: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    Ngrid: np.ndarray,
    h: np.ndarray,
    P: int,
    xi: float,
    eta: np.ndarray,
    rc: float,
    offset: np.ndarray,
    offsetxyz: np.ndarray,
    Hperp: np.ndarray,
    Hpara: np.ndarray,
    rvals: np.ndarray,
    real_precomp: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    grid_precomp: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    k_precomp: tuple[np.ndarray, np.ndarray] | None = None,
    slab_factor: float = 1.0,
) -> np.ndarray:
    """
    Compute the magnetic field at particle positions due to induced dipoles.

    This is a direct translation of MagneticField.m, with two optional
    (mathematically identical) speedups:

      * grid_precomp: uses precomputed Spread/Contract weights and indices
        (prepare_grid_kernel), avoiding Python loops inside GMRES.
      * k_precomp: uses precomputed reciprocal-space kernel data
        (precompute_kspace), avoiding rebuilding the k-grid inside GMRES.

    If these are not provided, the original loop-based Spread/Contract and
    on-the-fly k-grid construction are used.
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(m, dtype=np.complex128)
    lambdap = np.asarray(lambdap, dtype=np.float64).reshape(-1)
    box = np.asarray(box, dtype=np.float64).reshape(3)

    # --- Spread to grid ---
    if grid_precomp is None:
        Hx, Hy, Hz = spread(x, m, Ngrid, h, xi, eta, P, offset, offsetxyz)
        H = np.stack((Hx, Hy, Hz), axis=3)  # (Nx,Ny,Nz,3)
        w_contract = None
        linnode = None
    else:
        linnode, w_spread, w_contract = grid_precomp
        H = spread_fast(m, Ngrid, linnode, w_spread)

    # --- Fourier transform on each component (batched) ---
    fH = np.fft.fftshift(np.fft.fftn(H, axes=(0, 1, 2)), axes=(0, 1, 2))

    # --- Scale in reciprocal space ---
    if k_precomp is None:
        # Build k-grid exactly as MATLAB does (matched to fftshift ordering)
        Ngrid = np.asarray(Ngrid, dtype=int).reshape(3)
        Nx, Ny, Nz = int(Ngrid[0]), int(Ngrid[1]), int(Ngrid[2])
        kx = np.arange(-int(np.ceil((Nx - 1) / 2.0)), int(np.floor((Nx - 1) / 2.0)) + 1) * (2.0 * np.pi / box[0])
        ky = np.arange(-int(np.ceil((Ny - 1) / 2.0)), int(np.floor((Ny - 1) / 2.0)) + 1) * (2.0 * np.pi / box[1])
        kz = np.arange(-int(np.ceil((Nz - 1) / 2.0)), int(np.floor((Nz - 1) / 2.0)) + 1) * (2.0 * np.pi / box[2])
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        k = np.stack((KX, KY, KZ), axis=3)
        fHtilde = scale(fH, k, Ngrid, xi, eta)
    else:
        khat, Htildecoeff = k_precomp
        fHtilde = scale_fast(fH, khat, Htildecoeff)

    # --- Inverse FFT back to real space (batched) ---
    Htilde = np.fft.ifftn(np.fft.ifftshift(fHtilde, axes=(0, 1, 2)), axes=(0, 1, 2))

    # --- Contract grid to particle locations ---
    if grid_precomp is None:
        Hk = contract(x, Ngrid, h, xi, eta, P, Htilde, offset, offsetxyz)
    else:
        Hk = contract_fast(Htilde, Ngrid, linnode, w_contract)

    # --- Real-space contribution ---
    if real_precomp is None:
        Hr = real_space(x, m, lambdap, box, p1, p2, rc, Hperp, Hpara, rvals)
    else:
        p1f, p2f, rhat, A, B = real_precomp
        Hr = real_space_fast(m, lambdap, p1f, p2f, rhat, A, B, self_perp=float(Hperp[0]))

    # --- Slab correction (MATLAB code uses the no-4*pi version) ---
    H_slab = -float(slab_factor) * np.sum(m[:, 2]) / np.prod(box) * np.array([0.0, 0.0, 1.0], dtype=np.complex128)
    H_slab = np.tile(H_slab.reshape((1, 3)), (x.shape[0], 1))

    return Hk + Hr - H_slab


def magnetic_field_components(
    x: np.ndarray,
    m: np.ndarray,
    lambdap: np.ndarray,
    box: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    Ngrid: np.ndarray,
    h: np.ndarray,
    P: int,
    xi: float,
    eta: np.ndarray,
    rc: float,
    offset: np.ndarray,
    offsetxyz: np.ndarray,
    Hperp: np.ndarray,
    Hpara: np.ndarray,
    rvals: np.ndarray,
    real_precomp: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    grid_precomp: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    k_precomp: tuple[np.ndarray, np.ndarray] | None = None,
    slab_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Diagnostic helper: return the reciprocal-space (Hk), real-space (Hr),
    and slab-correction (H_slab) contributions separately.

    This is useful when you suspect xi-dependence: the *sum* Hk + Hr - H_slab
    should be (nearly) xi-invariant, while the individual pieces will shift
    with xi.

    Returns
    -------
    Hk : (N,3) complex
    Hr : (N,3) complex
    H_slab : (N,3) complex  (already replicated to particle shape)
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(m, dtype=np.complex128)
    lambdap = np.asarray(lambdap, dtype=np.float64).reshape(-1)
    box = np.asarray(box, dtype=np.float64).reshape(3)

    # --- Spread to grid ---
    if grid_precomp is None:
        Hx, Hy, Hz = spread(x, m, Ngrid, h, xi, eta, P, offset, offsetxyz)
        H = np.stack((Hx, Hy, Hz), axis=3)  # (Nx,Ny,Nz,3)
    else:
        linnode, w_spread, w_contract = grid_precomp
        H = spread_fast(m, Ngrid, linnode, w_spread)

    # --- Fourier transform (batched over vector component) ---
    fH = np.fft.fftshift(np.fft.fftn(H, axes=(0, 1, 2)), axes=(0, 1, 2))

    # --- Scale in reciprocal space ---
    if k_precomp is None:
        Ngrid = np.asarray(Ngrid, dtype=int).reshape(3)
        Nx, Ny, Nz = int(Ngrid[0]), int(Ngrid[1]), int(Ngrid[2])
        kx = np.arange(-int(np.ceil((Nx - 1) / 2.0)), int(np.floor((Nx - 1) / 2.0)) + 1) * (2.0 * np.pi / box[0])
        ky = np.arange(-int(np.ceil((Ny - 1) / 2.0)), int(np.floor((Ny - 1) / 2.0)) + 1) * (2.0 * np.pi / box[1])
        kz = np.arange(-int(np.ceil((Nz - 1) / 2.0)), int(np.floor((Nz - 1) / 2.0)) + 1) * (2.0 * np.pi / box[2])
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        k = np.stack((KX, KY, KZ), axis=3)
        fHtilde = scale(fH, k, Ngrid, xi, eta)
    else:
        khat, Htildecoeff = k_precomp
        fHtilde = scale_fast(fH, khat, Htildecoeff)

    # --- Inverse FFT ---
    Htilde = np.fft.ifftn(np.fft.ifftshift(fHtilde, axes=(0, 1, 2)), axes=(0, 1, 2))

    # --- Contract back to particles ---
    if grid_precomp is None:
        Hk = contract(x, Ngrid, h, xi, eta, P, Htilde, offset, offsetxyz)
    else:
        linnode, w_spread, w_contract = grid_precomp
        Hk = contract_fast(Htilde, Ngrid, linnode, w_contract)

    # --- Real-space contribution ---
    if real_precomp is None:
        Hr = real_space(x, m, lambdap, box, p1, p2, rc, Hperp, Hpara, rvals)
    else:
        p1f, p2f, rhat, A, B = real_precomp
        Hr = real_space_fast(m, lambdap, p1f, p2f, rhat, A, B, self_perp=float(Hperp[0]))

    # --- Slab correction ---
    H_slab = -float(slab_factor) * np.sum(m[:, 2]) / np.prod(box) * np.array([0.0, 0.0, 1.0], dtype=np.complex128)
    H_slab = np.tile(H_slab.reshape((1, 3)), (x.shape[0], 1))

    return Hk, Hr, H_slab
