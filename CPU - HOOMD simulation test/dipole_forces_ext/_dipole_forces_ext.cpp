// dipole_forces_ext.cpp
// Pybind11 extension for dipole-dipole forces with periodic x,y and optional periodic z.
//
// v0.5.x:
//   - Keep only one supported Python API (no legacy overloads).
//   - Optional periodicity in z (periodic_z=true by default) so real-space forces can
//     match fully-periodic simulations (HOOMD's default) while still allowing slab/open-z
//     approximations when desired.
//   - Optional cell-list acceleration (use_cell_list=true by default) to avoid O(N^2)
//     scaling for large N at finite cutoff.
//   - Sign convention: r_vec = r_i - r_j, so F_i = -∇_{r_i} U.
//
// Exposes:
//   compute_forces(pos, dip, box, r_cut, prefactor=1.0, periodic_z=True, use_cell_list=True) -> (forces, pe)
//
// forces: (N,3) float64
// pe:     (N,)  float64 per-particle potential energies, partitioned with 1/2 per pair.
//
// Notes:
// - Assumes an orthorhombic box given as [Lx, Ly, Lz].
// - Periodic boundary conditions are always applied in x and y (minimum image).
// - In z: periodic if periodic_z==True, else open (no wrapping).
// - For periodic dimensions, requires r_cut < 0.5*L_dim so the minimum-image convention
//   is unambiguous (standard for truncated pair potentials).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

static inline double wrap_min_image(double dx, double L)
{
    return dx - L * std::rint(dx / L);
}

struct Neighbor1D
{
    int n;
    int idx[3];
};

static inline Neighbor1D neighbor_cells_1d(int c, int Nc, bool periodic)
{
    Neighbor1D out;
    out.n = 0;

    for (int d = -1; d <= 1; ++d)
    {
        int cc = c + d;
        if (periodic)
        {
            cc %= Nc;
            if (cc < 0)
                cc += Nc;
        }
        else
        {
            if (cc < 0 || cc >= Nc)
                continue;
        }

        bool dup = false;
        for (int k = 0; k < out.n; ++k)
        {
            if (out.idx[k] == cc)
            {
                dup = true;
                break;
            }
        }
        if (!dup)
        {
            out.idx[out.n] = cc;
            out.n += 1;
        }
    }

    return out;
}

static inline int safe_cell_count(double L, double r_cut)
{
    // Choose number of cells so that cell size h = L / Nc >= r_cut.
    // This guarantees that all neighbors within r_cut are in the current cell
    // or an adjacent cell (delta in {-1,0,1}).
    int Nc = static_cast<int>(std::floor(L / r_cut));
    if (Nc < 1)
        Nc = 1;
    return Nc;
}

static inline int cell_index(double x_centered, double L, int Nc)
{
    // Convert HOOMD-style centered coordinates in [-L/2, L/2) to [0, L)
    // and compute 0-based cell index.
    double x = x_centered + 0.5 * L;
    // Wrap to [0, L) (safe even if already inside).
    x -= L * std::floor(x / L);

    const double h = L / static_cast<double>(Nc);
    int ix = static_cast<int>(std::floor(x / h));
    if (ix < 0)
        ix = 0;
    if (ix >= Nc)
        ix = Nc - 1;
    return ix;
}

static std::pair<py::array_t<double>, py::array_t<double>>
compute_forces_bruteforce(py::array_t<double, py::array::c_style | py::array::forcecast> pos,
                          py::array_t<double, py::array::c_style | py::array::forcecast> dip,
                          py::array_t<double, py::array::c_style | py::array::forcecast> box,
                          double r_cut,
                          double prefactor,
                          bool periodic_z)
{
    auto p = pos.unchecked<2>();
    auto m = dip.unchecked<2>();
    auto b = box.unchecked<1>();

    const ssize_t N = p.shape(0);
    if (p.shape(1) != 3 || m.shape(0) != N || m.shape(1) != 3 || b.shape(0) != 3)
        throw std::runtime_error("Invalid shapes: pos(N,3), dip(N,3), box(3) required.");

    const double Lx = b(0);
    const double Ly = b(1);
    const double Lz = b(2);
    if (!(Lx > 0.0 && Ly > 0.0 && Lz > 0.0))
        throw std::runtime_error("Box lengths must be positive.");

    if (!(r_cut > 0.0))
        throw std::runtime_error("r_cut must be > 0.");

    // Minimum-image validity checks.
    const double half_x = 0.5 * Lx;
    const double half_y = 0.5 * Ly;
    if (r_cut >= std::min(half_x, half_y))
        throw std::runtime_error("r_cut too large for minimum-image periodic x,y. Require r_cut < 0.5*min(Lx,Ly).");

    if (periodic_z)
    {
        const double half_z = 0.5 * Lz;
        if (r_cut >= half_z)
            throw std::runtime_error("r_cut too large for minimum-image periodic z. Require r_cut < 0.5*Lz when periodic_z=True.");
    }

    py::array_t<double> forces({N, (ssize_t)3});
    py::array_t<double> pe({N});

    auto f = forces.mutable_unchecked<2>();
    auto e = pe.mutable_unchecked<1>();

    for (ssize_t i = 0; i < N; ++i)
    {
        f(i, 0) = 0.0;
        f(i, 1) = 0.0;
        f(i, 2) = 0.0;
        e(i) = 0.0;
    }

    const double r_cut2 = r_cut * r_cut;

#pragma omp parallel for schedule(static)
    for (ssize_t i = 0; i < N; ++i)
    {
        double fx_i = 0.0, fy_i = 0.0, fz_i = 0.0;
        double ei_i = 0.0;

        const double xi = p(i, 0);
        const double yi = p(i, 1);
        const double zi = p(i, 2);

        const double mux_i = m(i, 0);
        const double muy_i = m(i, 1);
        const double muz_i = m(i, 2);

        for (ssize_t j = i + 1; j < N; ++j)
        {
            // r_vec = r_i - r_j
            double dx = wrap_min_image(xi - p(j, 0), Lx);
            double dy = wrap_min_image(yi - p(j, 1), Ly);
            double dz = zi - p(j, 2);
            if (periodic_z)
                dz = wrap_min_image(dz, Lz);

            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 > r_cut2 || r2 < 1e-24)
                continue;

            const double inv_r2 = 1.0 / r2;
            const double inv_r = std::sqrt(inv_r2);
            const double inv_r3 = inv_r2 * inv_r;
            const double inv_r5 = inv_r3 * inv_r2;

            const double mux_j = m(j, 0);
            const double muy_j = m(j, 1);
            const double muz_j = m(j, 2);

            const double mi_mj = mux_i * mux_j + muy_i * muy_j + muz_i * muz_j;
            const double mi_r = mux_i * dx + muy_i * dy + muz_i * dz;
            const double mj_r = mux_j * dx + muy_j * dy + muz_j * dz;

            // Pair energy: U = prefactor * [ (mi·mj)/r^3 - 3 (mi·r)(mj·r)/r^5 ]
            const double Upair = prefactor * (mi_mj * inv_r3 - 3.0 * (mi_r * mj_r) * inv_r5);

            // Force on i due to j for r_vec = r_i - r_j:
            // F_i = (3*pref/r^5) * [ (mi·mj) r + (mj·r) mi + (mi·r) mj - 5(mi·r)(mj·r) r / r^2 ]
            const double coef = prefactor * 3.0 * inv_r5;

            const double fx = coef * (mi_mj * dx + mj_r * mux_i + mi_r * mux_j - 5.0 * (mi_r * mj_r) * dx * inv_r2);
            const double fy = coef * (mi_mj * dy + mj_r * muy_i + mi_r * muy_j - 5.0 * (mi_r * mj_r) * dy * inv_r2);
            const double fz = coef * (mi_mj * dz + mj_r * muz_i + mi_r * muz_j - 5.0 * (mi_r * mj_r) * dz * inv_r2);

            fx_i += fx;
            fy_i += fy;
            fz_i += fz;
            ei_i += 0.5 * Upair;

#pragma omp atomic
            f(j, 0) -= fx;
#pragma omp atomic
            f(j, 1) -= fy;
#pragma omp atomic
            f(j, 2) -= fz;
#pragma omp atomic
            e(j) += 0.5 * Upair;
        }

#pragma omp atomic
        f(i, 0) += fx_i;
#pragma omp atomic
        f(i, 1) += fy_i;
#pragma omp atomic
        f(i, 2) += fz_i;
#pragma omp atomic
        e(i) += ei_i;
    }

    return {forces, pe};
}

static std::pair<py::array_t<double>, py::array_t<double>>
compute_forces_celllist(py::array_t<double, py::array::c_style | py::array::forcecast> pos,
                        py::array_t<double, py::array::c_style | py::array::forcecast> dip,
                        py::array_t<double, py::array::c_style | py::array::forcecast> box,
                        double r_cut,
                        double prefactor,
                        bool periodic_z)
{
    auto p = pos.unchecked<2>();
    auto m = dip.unchecked<2>();
    auto b = box.unchecked<1>();

    const ssize_t N = p.shape(0);
    if (p.shape(1) != 3 || m.shape(0) != N || m.shape(1) != 3 || b.shape(0) != 3)
        throw std::runtime_error("Invalid shapes: pos(N,3), dip(N,3), box(3) required.");

    const double Lx = b(0);
    const double Ly = b(1);
    const double Lz = b(2);
    if (!(Lx > 0.0 && Ly > 0.0 && Lz > 0.0))
        throw std::runtime_error("Box lengths must be positive.");

    if (!(r_cut > 0.0))
        throw std::runtime_error("r_cut must be > 0.");

    // Minimum-image validity checks.
    const double half_x = 0.5 * Lx;
    const double half_y = 0.5 * Ly;
    if (r_cut >= std::min(half_x, half_y))
        throw std::runtime_error("r_cut too large for minimum-image periodic x,y. Require r_cut < 0.5*min(Lx,Ly).");

    if (periodic_z)
    {
        const double half_z = 0.5 * Lz;
        if (r_cut >= half_z)
            throw std::runtime_error("r_cut too large for minimum-image periodic z. Require r_cut < 0.5*Lz when periodic_z=True.");
    }

    py::array_t<double> forces({N, (ssize_t)3});
    py::array_t<double> pe({N});

    auto f = forces.mutable_unchecked<2>();
    auto e = pe.mutable_unchecked<1>();

    for (ssize_t i = 0; i < N; ++i)
    {
        f(i, 0) = 0.0;
        f(i, 1) = 0.0;
        f(i, 2) = 0.0;
        e(i) = 0.0;
    }

    const double r_cut2 = r_cut * r_cut;

    // Build cell list.
    const int Nx = safe_cell_count(Lx, r_cut);
    const int Ny = safe_cell_count(Ly, r_cut);
    const int Nz = safe_cell_count(Lz, r_cut);

    const int n_cells = Nx * Ny * Nz;
    std::vector<int> head(n_cells, -1);
    std::vector<int> next(static_cast<size_t>(N), -1);

    std::vector<int> cellx(static_cast<size_t>(N));
    std::vector<int> celly(static_cast<size_t>(N));
    std::vector<int> cellz(static_cast<size_t>(N));

    for (ssize_t i = 0; i < N; ++i)
    {
        int ix = cell_index(p(i, 0), Lx, Nx);
        int iy = cell_index(p(i, 1), Ly, Ny);
        int iz = cell_index(p(i, 2), Lz, Nz);

        cellx[static_cast<size_t>(i)] = ix;
        celly[static_cast<size_t>(i)] = iy;
        cellz[static_cast<size_t>(i)] = iz;

        const int cid = (ix * Ny + iy) * Nz + iz;
        next[static_cast<size_t>(i)] = head[cid];
        head[cid] = static_cast<int>(i);
    }

#pragma omp parallel for schedule(static)
    for (ssize_t i = 0; i < N; ++i)
    {
        double fx_i = 0.0, fy_i = 0.0, fz_i = 0.0;
        double ei_i = 0.0;

        const double xi = p(i, 0);
        const double yi = p(i, 1);
        const double zi = p(i, 2);

        const double mux_i = m(i, 0);
        const double muy_i = m(i, 1);
        const double muz_i = m(i, 2);

        const int ix = cellx[static_cast<size_t>(i)];
        const int iy = celly[static_cast<size_t>(i)];
        const int iz = cellz[static_cast<size_t>(i)];

        // Neighbor cells (avoid duplicates when Nc is 1 or 2).
        const Neighbor1D nx = neighbor_cells_1d(ix, Nx, true);
        const Neighbor1D ny = neighbor_cells_1d(iy, Ny, true);
        const Neighbor1D nz = neighbor_cells_1d(iz, Nz, periodic_z);

        for (int ax = 0; ax < nx.n; ++ax)
        {
            const int jx = nx.idx[ax];
            for (int ay = 0; ay < ny.n; ++ay)
            {
                const int jy = ny.idx[ay];
                for (int az = 0; az < nz.n; ++az)
                {
                    const int jz = nz.idx[az];
                    const int cid = (jx * Ny + jy) * Nz + jz;

                    for (int j = head[cid]; j != -1; j = next[static_cast<size_t>(j)])
                    {
                        if (j <= i)
                            continue;

                        // r_vec = r_i - r_j
                        double dx = wrap_min_image(xi - p(j, 0), Lx);
                        double dy = wrap_min_image(yi - p(j, 1), Ly);
                        double dz = zi - p(j, 2);
                        if (periodic_z)
                            dz = wrap_min_image(dz, Lz);

                        const double r2 = dx * dx + dy * dy + dz * dz;
                        if (r2 > r_cut2 || r2 < 1e-24)
                            continue;

                        const double inv_r2 = 1.0 / r2;
                        const double inv_r = std::sqrt(inv_r2);
                        const double inv_r3 = inv_r2 * inv_r;
                        const double inv_r5 = inv_r3 * inv_r2;

                        const double mux_j = m(j, 0);
                        const double muy_j = m(j, 1);
                        const double muz_j = m(j, 2);

                        const double mi_mj = mux_i * mux_j + muy_i * muy_j + muz_i * muz_j;
                        const double mi_r = mux_i * dx + muy_i * dy + muz_i * dz;
                        const double mj_r = mux_j * dx + muy_j * dy + muz_j * dz;

                        const double Upair = prefactor * (mi_mj * inv_r3 - 3.0 * (mi_r * mj_r) * inv_r5);

                        const double coef = prefactor * 3.0 * inv_r5;

                        const double fx = coef * (mi_mj * dx + mj_r * mux_i + mi_r * mux_j - 5.0 * (mi_r * mj_r) * dx * inv_r2);
                        const double fy = coef * (mi_mj * dy + mj_r * muy_i + mi_r * muy_j - 5.0 * (mi_r * mj_r) * dy * inv_r2);
                        const double fz = coef * (mi_mj * dz + mj_r * muz_i + mi_r * muz_j - 5.0 * (mi_r * mj_r) * dz * inv_r2);

                        fx_i += fx;
                        fy_i += fy;
                        fz_i += fz;
                        ei_i += 0.5 * Upair;

#pragma omp atomic
                        f(j, 0) -= fx;
#pragma omp atomic
                        f(j, 1) -= fy;
#pragma omp atomic
                        f(j, 2) -= fz;
#pragma omp atomic
                        e(j) += 0.5 * Upair;
                    }
                }
            }
        }

#pragma omp atomic
        f(i, 0) += fx_i;
#pragma omp atomic
        f(i, 1) += fy_i;
#pragma omp atomic
        f(i, 2) += fz_i;
#pragma omp atomic
        e(i) += ei_i;
    }

    return {forces, pe};
}

// ------------------------------
// Public API (pybind)
// ------------------------------
static std::pair<py::array_t<double>, py::array_t<double>>
compute_forces_core(py::array_t<double, py::array::c_style | py::array::forcecast> pos,
                    py::array_t<double, py::array::c_style | py::array::forcecast> dip,
                    py::array_t<double, py::array::c_style | py::array::forcecast> box,
                    double r_cut,
                    double prefactor,
                    bool periodic_z,
                    bool use_cell_list)
{
    if (use_cell_list)
        return compute_forces_celllist(pos, dip, box, r_cut, prefactor, periodic_z);
    else
        return compute_forces_bruteforce(pos, dip, box, r_cut, prefactor, periodic_z);
}

PYBIND11_MODULE(_dipole_forces_ext, m)
{
    m.doc() = "Dipole-dipole forces with periodic x,y and optional periodic z.";
    m.attr("__version__") = "0.5.1";

    // Primary signature
    m.def("compute_forces",
          &compute_forces_core,
          py::arg("pos"),
          py::arg("dip"),
          py::arg("box"),
          py::arg("r_cut"),
          py::arg("prefactor") = 1.0,
          py::arg("periodic_z") = true,
          py::arg("use_cell_list") = true,
          R"pbdoc(
Compute dipole-dipole forces and per-particle energies.

Parameters
----------
pos : (N,3) float64
    Particle positions in a centered orthorhombic box (HOOMD convention).
dip : (N,3) float64
    Dipole vectors m_i.
box : (3,) float64
    Box lengths [Lx, Ly, Lz].
r_cut : float
    Real-space cutoff. For periodic dimensions, must satisfy r_cut < 0.5*L_dim.
prefactor : float
    Multiplicative prefactor for the interaction.
periodic_z : bool
    If True, apply minimum-image periodicity in z.
    If False, treat z as open (no wrapping).
use_cell_list : bool
    If True, use an internal cell list (O(N)) for neighbor search at finite r_cut.
    If False, use an O(N^2) double loop.

Returns
-------
forces : (N,3) float64
pe : (N,) float64
    Per-particle energies (0.5*U_pair accumulated on each particle).
)pbdoc");
}
