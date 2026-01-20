#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diagnostics_force_series_plots_v6_norm2R_driver.py

Driver-compatible HOOMD v6.x diagnostic for validating:
  1) Mutual polarization dipoles from capacitance.run_capacitance (eps_p from beta).
  2) Dipolar forces/energies from dipole_forces_ext (C++ extension) inside HOOMD.
  3) Head-to-tail and side-by-side series vs separation.
  4) Normalized plots vs constant-dipole analytic expectation using normalization at r_ref = 2R.
  5) Extra plots: raw |F|, |U| (log-log), ratio vs pair-analytic, dipole enhancement.
  6) Kernel tests (fixed dipoles) inside HOOMD, including energy-force finite-difference.

Prerequisites (typical):
  - A HOOMD v6 environment.
  - dipole_forces_ext built in the same directory (or importable on PYTHONPATH).
    Build command (from the directory containing setup_dipole_forces_ext.py):
        python setup_dipole_forces_ext.py build_ext --inplace

Outputs (in out_force_series/):
  mutual_polarization_force_energy_series.csv
  plot_head_force_norm_2R.png
  plot_head_energy_norm_2R.png
  plot_side_force_norm_2R.png
  plot_side_energy_norm_2R.png
  plot_head_force_raw_mag.png
  plot_head_energy_raw_mag.png
  plot_side_force_raw_mag.png
  plot_side_energy_raw_mag.png
  plot_ratio_pairF.png
  plot_ratio_pairU.png
  plot_m_enhancement.png

"""

from __future__ import annotations

import sys
import csv
import subprocess
from pathlib import Path
import importlib
from typing import List, Dict, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size

import hoomd
import hoomd.md


# ----------------------------
# Plot styling (match Code 2)
# ----------------------------

mpl.rcdefaults()
mpl.rcParams.update(mpl.rcParamsDefault)

mpl.rcParams["text.usetex"] = False
mpl.rcParams["font.family"] = "serif"

PLOT_LW = 3
MARKER_SIZE = 14
TICK_FONTSIZE = 36
AXES_FONTSIZE = 48
LEGEND_FONTSIZE = 24

mpl.rcParams["axes.linewidth"] = PLOT_LW
mpl.rcParams["savefig.bbox"] = None
mpl.rcParams["figure.constrained_layout.use"] = False

# Treat figsize as desired *plot area* size (inches)
PLOT_AREA_W_IN = 10.0
PLOT_AREA_H_IN = 9.0
LOCK_MIN_MARGINS_IN = (1.3, 0.5, 1.2, 0.6)  # L, R, B, T in inches

ZORDER_REF = 1.05
ZORDER_ANALYTIC = 1.10
ZORDER_DATA = 3.00


def lock_plot_area(ax, plot_width_in, plot_height_in,
                   min_margins=LOCK_MIN_MARGINS_IN,  # L, R, B, T (inches)
                   max_iter=2):
    """Make the axes' plotting area exactly (plot_width_in × plot_height_in) inches."""
    fig = ax.figure

    def _measure_needed_margins():
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bb_ax = ax.bbox
        try:
            bb_tight = ax.get_tightbbox(renderer, call_axes_locator=True)
        except TypeError:
            bb_tight = ax.get_tightbbox(renderer)
        dpi = fig.dpi
        left = max((bb_ax.x0 - bb_tight.x0) / dpi, 0.0)
        right = max((bb_tight.x1 - bb_ax.x1) / dpi, 0.0)
        bottom = max((bb_ax.y0 - bb_tight.y0) / dpi, 0.0)
        top = max((bb_tight.y1 - bb_ax.y1) / dpi, 0.0)
        return left, right, bottom, top

    left, right, bottom, top = min_margins
    for _ in range(max_iter):
        fig.set_size_inches(plot_width_in + left + right,
                            plot_height_in + bottom + top,
                            forward=True)

        h = [Size.Fixed(left), Size.Fixed(plot_width_in), Size.Fixed(right)]
        v = [Size.Fixed(bottom), Size.Fixed(plot_height_in), Size.Fixed(top)]
        divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
        ax.set_axes_locator(divider.new_locator(nx=1, ny=1))

        need_left, need_right, need_bottom, need_top = _measure_needed_margins()
        new = (max(need_left, min_margins[0]),
               max(need_right, min_margins[1]),
               max(need_bottom, min_margins[2]),
               max(need_top, min_margins[3]))
        if new == (left, right, bottom, top):
            break
        left, right, bottom, top = new

    return left, right, bottom, top


def _style_axes(ax):
    ax.tick_params(which="both", direction="in", width=PLOT_LW, length=14)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE, pad=10)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, pad=10)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_LW)


def _new_fig_ax():
    fig, ax = plt.subplots(figsize=(PLOT_AREA_W_IN, PLOT_AREA_H_IN))
    return fig, ax


def _beta_tex(beta: float) -> str:
    return rf"$\beta={float(beta):g}$"


def _beta_style(beta: float):
    """Return (color, marker, markersize) using the palette conventions."""
    b = float(beta)
    if np.isclose(b, 1.0):
        return "#1E88E5", "s", MARKER_SIZE - 1
    if abs(b) < 0.05:
        return "#004D40", "^", MARKER_SIZE
    if b < 0.0:
        return "#D81B60", "o", MARKER_SIZE

    palette = ["#D81B60", "#004D40", "#1E88E5", "#6A1B9A", "#F9A825"]
    idx = int(abs(hash(round(b, 6))) % len(palette))
    return palette[idx], "o", MARKER_SIZE


def _plot_data_series(ax, x, y, color, marker, markersize, label: str):
    ax.plot(
        x, y,
        color=color, linewidth=PLOT_LW,
        marker=marker, markersize=markersize,
        markeredgecolor=color, markerfacecolor="white", markeredgewidth=PLOT_LW,
        linestyle=":",
        zorder=ZORDER_DATA,
        label=label,
    )


def _plot_analytic(ax, x, y, color, label: str):
    ax.plot(
        x, y,
        color=color, linewidth=PLOT_LW,
        linestyle="-",
        zorder=ZORDER_ANALYTIC,
        label=label,
    )


def _savefig(fig, ax, path: Path, dpi: int = 300):
    _style_axes(ax)
    lock_plot_area(ax, PLOT_AREA_W_IN, PLOT_AREA_H_IN,
                   min_margins=LOCK_MIN_MARGINS_IN, max_iter=2)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    print("[plots]", path, flush=True)


# ----------------------------
# Parameters (match driver defaults unless you intentionally override)
# ----------------------------

OUTDIR = Path("out_force_series")
OUTDIR.mkdir(exist_ok=True)

# HOOMD measurement box (cubic)
BOX_L = 80.0
BOX = np.array([BOX_L, BOX_L, BOX_L], dtype=np.float64)

# Dipole kernel parameters
R_CUT_USER = 25.0
PREF = 1.0

# Capacitance parameters
XI = 0.5
ERRTOL = 1.0e-5
SLAB_FACTOR = 1.0
USE_SCIPY_GMRES = True

# Field direction for diagnostics
HHAT = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# Dipole-force boundary condition controls (matches the v0.4.x extension)
# - Keep periodic_z=True if you want real-space forces to match HOOMD's default 3D periodicity
#   even when using a slab_factor in the reciprocal-space dipole solve.
DIPOLE_PERIODIC_Z = True
DIPOLE_USE_CELL_LIST = True

# Optional: attempt to build dipole_forces_ext in-place if import fails.
AUTO_BUILD_DIPOLE_FORCES_EXT = False

# Separations to sample (center-center distance)
R_LIST = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

# beta values to compare
BETA_TARGETS = [0.01, 1.0]

# "Nearly-static" evaluation step in HOOMD
DT_EVAL = 1.0e-12
GAMMA_EVAL = 1.0e12
KT_EVAL = 0.0

# Normalization reference: r_ref = 2R
PARTICLE_RADIUS = 1.0
R_REF = 2.0 * float(PARTICLE_RADIUS)

# Finite-difference step for energy-force validation
FD_DR = 1.0e-4

# For this diagnostic, positions change between evaluations. Use pol_every_steps=1
# so dipoles always reflect the current configuration.
POL_EVERY_STEPS_DIAG = 1



# ----------------------------
# dipole_forces_ext package import (compiled kernel + pure-Python ewald/capacitance)
# ----------------------------


def _import_or_build_dipole_forces_ext_package():
    """Import dipole_forces_ext, optionally building it in-place if missing."""
    try:
        import importlib
        return importlib.import_module("dipole_forces_ext")
    except Exception as e:
        if not AUTO_BUILD_DIPOLE_FORCES_EXT:
            raise ImportError(
                "Could not import dipole_forces_ext. Build it with:\n"
                "  python setup_dipole_forces_ext.py build_ext --inplace\n"
                "in the same directory as this script."
            ) from e

        here = Path(__file__).resolve().parent
        setup_path = here / "setup_dipole_forces_ext.py"
        if not setup_path.exists():
            raise ImportError(
                "AUTO_BUILD_DIPOLE_FORCES_EXT=True, but setup_dipole_forces_ext.py was not found next to this script."
            ) from e

        cmd = [sys.executable, str(setup_path), "build_ext", "--inplace"]
        try:
            subprocess.check_call(cmd, cwd=str(here))
        except Exception as build_e:
            raise ImportError(
                "Automatic build of dipole_forces_ext failed. Try building manually with:\n"
                "  python setup_dipole_forces_ext.py build_ext --inplace"
            ) from build_e

        import importlib
        return importlib.import_module("dipole_forces_ext")


dipole_forces_ext = _import_or_build_dipole_forces_ext_package()
from dipole_forces_ext import capacitance


print("HOOMD version:", hoomd.version.version, flush=True)

print("[diag] dipole_forces_ext version:", getattr(dipole_forces_ext, "__version__", "unknown"), flush=True)

# r_cut safety: minimum-image requires r_cut < 0.5*L_dim for periodic dimensions.
r_cut_max_xy = 0.499 * float(min(BOX[0], BOX[1]))
r_cut_max_z = (0.499 * float(BOX[2])) if DIPOLE_PERIODIC_Z else float("inf")
r_cut_max = float(min(r_cut_max_xy, r_cut_max_z))
R_CUT = float(min(R_CUT_USER, r_cut_max))

print(
    f"[params] BOX_L={BOX_L:g} r_cut={R_CUT:g} (user={R_CUT_USER:g}, max={r_cut_max:g}) pref={PREF:g}",
    flush=True,
)
print(
    f"[cap] xi={XI:g} errtol={ERRTOL:g} slab_factor={SLAB_FACTOR:g} scipy_gmres={USE_SCIPY_GMRES}",
    flush=True,
)
print(f"[geom] HHAT={HHAT.tolist()}  |HHAT|={float(np.linalg.norm(HHAT)):.6g}", flush=True)
print(f"[dipole-force] periodic_z={DIPOLE_PERIODIC_Z} use_cell_list={DIPOLE_USE_CELL_LIST}", flush=True)
print(f"[norm] PARTICLE_RADIUS={PARTICLE_RADIUS:g} -> r_ref=2R={R_REF:g}", flush=True)
print(f"[eval] dt={DT_EVAL:g} gamma={GAMMA_EVAL:g} kT={KT_EVAL:g}", flush=True)


# ----------------------------
# Helpers: beta <-> eps_p and point-dipole analytics
# ----------------------------


def eps_from_beta(beta_target: float) -> float:
    """Invert beta = (eps-1)/(eps+2) -> eps = (1+2beta)/(1-beta)."""
    b = float(beta_target)
    if b >= 0.999999:
        return 1.0e6
    if b <= -0.999999:
        return 1.0e-6
    return (1.0 + 2.0 * b) / (1.0 - b)


def beta_eff_from_eps(eps_p: float) -> float:
    e = float(eps_p)
    return (e - 1.0) / (e + 2.0)


def point_dipole_U(pref: float, m: float, r: float, mode: str) -> float:
    """Pair energy for two identical dipoles m aligned with HHAT."""
    pref = float(pref)
    m = float(m)
    r = float(r)
    if mode == "head":
        return -2.0 * pref * (m * m) / (r ** 3)
    if mode == "side":
        return +1.0 * pref * (m * m) / (r ** 3)
    raise ValueError("mode must be 'head' or 'side'")


def point_dipole_F(pref: float, m: float, r: float, mode: str) -> float:
    """Force component on particle at +r/2 along the separation axis."""
    pref = float(pref)
    m = float(m)
    r = float(r)
    if mode == "head":
        return -6.0 * pref * (m * m) / (r ** 4)
    if mode == "side":
        return +3.0 * pref * (m * m) / (r ** 4)
    raise ValueError("mode must be 'head' or 'side'")


# ----------------------------
# ----------------------------
# HOOMD Custom Forces
# ----------------------------

# Use the canonical implementations from the dipole_forces_ext package.
from dipole_forces_ext.hoomd_force import MutualPolarizationDipoleForceCached, FixedDipoleForce

# For this diagnostic, positions change between evaluations. Use pol_every_steps=1
# so dipoles always reflect the current configuration.
POL_EVERY_STEPS_DIAG = 1


# Static evaluator (reuse HOOMD sim; set positions; run 1 step)
# ----------------------------


class StaticEvaluator:
    def __init__(self, N: int, force_obj: hoomd.md.force.Custom, seed: int = 1):
        self.N = int(N)
        self.force_obj = force_obj

        self.device = hoomd.device.CPU()
        if getattr(self.device, "communicator", None) is not None:
            nr = int(self.device.communicator.num_ranks)
            if nr != 1:
                raise RuntimeError(f"This script requires 1 MPI rank. Found num_ranks={nr}")

        self.sim = hoomd.Simulation(device=self.device, seed=int(seed))

        snap = hoomd.Snapshot()
        snap.configuration.box = [float(BOX[0]), float(BOX[1]), float(BOX[2]), 0, 0, 0]

        if snap.communicator.rank == 0:
            snap.particles.N = self.N
            snap.particles.types = ["A"]
            snap.particles.typeid[:] = 0
            snap.particles.position[:] = np.zeros((self.N, 3), dtype=np.float32)

        self.sim.create_state_from_snapshot(snap)

        brown = hoomd.md.methods.Brownian(filter=hoomd.filter.All(), kT=float(KT_EVAL))
        brown.gamma["A"] = float(GAMMA_EVAL)

        integrator = hoomd.md.Integrator(dt=float(DT_EVAL), methods=[brown], forces=[self.force_obj])
        integrator.integrate_rotational_dof = False
        self.sim.operations.integrator = integrator

    def eval(self, positions: np.ndarray):
        positions = np.asarray(positions, dtype=np.float32)
        if positions.shape != (self.N, 3):
            raise ValueError(f"positions must have shape {(self.N, 3)}")

        with self.sim.state.cpu_local_snapshot as snap:
            snap.particles.position[:] = positions

        self.sim.run(1)

        with self.sim.state.cpu_local_snapshot as snap:
            pos = np.asarray(snap.particles.position, dtype=np.float64).copy()
            netF = np.asarray(snap.particles.net_force, dtype=np.float64).copy()

        if getattr(self.force_obj, "last_forces", None) is None:
            raise RuntimeError("Force object did not store last_forces. CustomForce may not have executed.")
        f = np.asarray(self.force_obj.last_forces, dtype=np.float64).copy()

        e_obj = getattr(self.force_obj, "last_energy", None)
        if e_obj is None:
            e = np.zeros((self.N,), dtype=np.float64)
        else:
            e = np.asarray(e_obj, dtype=np.float64).copy().reshape(-1)

        max_abs = float(np.max(np.abs(netF - f)))
        if max_abs > 1e-9:
            raise RuntimeError(f"HOOMD net_force != custom last_forces (max_abs={max_abs:.3e}).")

        fsum = np.sum(f, axis=0)
        if float(np.linalg.norm(fsum)) > 1e-9:
            print(f"[warn] net force sum is not ~0: |sumF|={float(np.linalg.norm(fsum)):.3e}", flush=True)

        return pos, f, e


# ----------------------------
# Measurements
# ----------------------------


def measure_single_dipole(eps_p: float) -> float:
    """Return m_single = dipole projection along HHAT for one particle under unit field_strength."""
    force = MutualPolarizationDipoleForceCached(
        eps_p=eps_p,
        xi=XI,
        errtol=ERRTOL,
        slab_factor=SLAB_FACTOR,
        Hhat=HHAT,
        r_cut=R_CUT,
        prefactor=PREF,
        field_strength=1.0,
        pol_every_steps=POL_EVERY_STEPS_DIAG,
        use_scipy_gmres=USE_SCIPY_GMRES,
        periodic_z=DIPOLE_PERIODIC_Z,
        use_cell_list=DIPOLE_USE_CELL_LIST,
    )
    ev = StaticEvaluator(N=1, force_obj=force, seed=11)
    ev.eval(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))

    if force.calls < 1 or force.last_dipoles is None:
        raise RuntimeError("MutualPolarizationDipoleForceCached did not execute for single particle.")
    m = float(force.last_dipoles[0].dot(HHAT))
    return m


def measure_pair_series(eps_p: float, mode: str):
    """Measure F and U across R_LIST for a fixed eps_p and geometry."""
    force = MutualPolarizationDipoleForceCached(
        eps_p=eps_p,
        xi=XI,
        errtol=ERRTOL,
        slab_factor=SLAB_FACTOR,
        Hhat=HHAT,
        r_cut=R_CUT,
        prefactor=PREF,
        field_strength=1.0,
        pol_every_steps=POL_EVERY_STEPS_DIAG,
        use_scipy_gmres=USE_SCIPY_GMRES,
        periodic_z=DIPOLE_PERIODIC_Z,
        use_cell_list=DIPOLE_USE_CELL_LIST,
    )
    ev = StaticEvaluator(N=2, force_obj=force, seed=22)

    out = []
    for r in R_LIST:
        r = float(r)
        if mode == "head":
            pos = np.array([[0.0, 0.0, -0.5 * r],
                            [0.0, 0.0, +0.5 * r]], dtype=np.float64)
            comp_axis = 2
        elif mode == "side":
            pos = np.array([[-0.5 * r, 0.0, 0.0],
                            [+0.5 * r, 0.0, 0.0]], dtype=np.float64)
            comp_axis = 0
        else:
            raise ValueError("mode must be 'head' or 'side'")

        pos_out, f, e = ev.eval(pos)

        r_meas = float(np.linalg.norm(pos_out[1] - pos_out[0]))
        f_comp = float(f[1, comp_axis])
        U_sum = float(np.sum(e))

        if force.last_dipoles is None:
            raise RuntimeError("No dipoles stored.")
        m0 = float(force.last_dipoles[0].dot(HHAT))
        m1 = float(force.last_dipoles[1].dot(HHAT))

        out.append(dict(
            r=r_meas,
            f_comp=f_comp,
            U_sum=U_sum,
            m0=m0,
            m1=m1,
            calls=int(force.calls),
        ))

    return out


# ----------------------------
# Plots
# ----------------------------


def _dense_x(min_x: float, max_x: float, n: int = 400) -> np.ndarray:
    return np.linspace(float(min_x), float(max_x), int(n), dtype=np.float64)


def make_plots(records: List[Dict], beta_targets: List[float]):
    def sel(beta: float, mode: str) -> List[Dict]:
        return [r for r in records if (abs(r["beta_target"] - beta) < 1e-12 and r["mode"] == mode)]

    rmin = float(min(rr["r"] for rr in records))
    rmax = float(max(rr["r"] for rr in records))
    xd = _dense_x(rmin, rmax, n=500)

    Fnorm_const = (float(R_REF) / xd) ** 4
    Unorm_const = (float(R_REF) / xd) ** 3

    xlab = r"$r_{ij}$"

    # 1) Normalized head force
    fig, ax = _new_fig_ax()
    for beta in beta_targets:
        rows = sel(beta, "head")
        x = np.array([rr["r"] for rr in rows], dtype=np.float64)
        y = np.array([rr["F_norm_2R"] for rr in rows], dtype=np.float64)
        c, mk, ms = _beta_style(beta)
        _plot_data_series(ax, x, y, c, mk, ms, label=_beta_tex(beta))
    _plot_analytic(ax, xd, Fnorm_const, "black", label=rf"const $m$: $(r_{{ref}}/r)^4$, $r_{{ref}}={R_REF:g}$")
    ax.set_xlabel(xlab, fontsize=AXES_FONTSIZE, labelpad=12)
    ax.set_ylabel(r"$F/F_{\mathrm{ref}}$", fontsize=AXES_FONTSIZE, labelpad=16)

    ax.legend(frameon=False, prop={"size": LEGEND_FONTSIZE}, loc="best")
    _savefig(fig, ax, OUTDIR / "plot_head_force_norm_2R.png")

    # 2) Normalized head energy
    fig, ax = _new_fig_ax()
    for beta in beta_targets:
        rows = sel(beta, "head")
        x = np.array([rr["r"] for rr in rows], dtype=np.float64)
        y = np.array([rr["U_norm_2R"] for rr in rows], dtype=np.float64)
        c, mk, ms = _beta_style(beta)
        _plot_data_series(ax, x, y, c, mk, ms, label=_beta_tex(beta))
    _plot_analytic(ax, xd, Unorm_const, "black", label=rf"const $m$: $(r_{{ref}}/r)^3$, $r_{{ref}}={R_REF:g}$")
    ax.set_xlabel(xlab, fontsize=AXES_FONTSIZE, labelpad=12)
    ax.set_ylabel(r"$U/U_{\mathrm{ref}}$", fontsize=AXES_FONTSIZE, labelpad=16)

    ax.legend(frameon=False, prop={"size": LEGEND_FONTSIZE}, loc="best")
    _savefig(fig, ax, OUTDIR / "plot_head_energy_norm_2R.png")

    # 3) Normalized side force
    fig, ax = _new_fig_ax()
    for beta in beta_targets:
        rows = sel(beta, "side")
        x = np.array([rr["r"] for rr in rows], dtype=np.float64)
        y = np.array([rr["F_norm_2R"] for rr in rows], dtype=np.float64)
        c, mk, ms = _beta_style(beta)
        _plot_data_series(ax, x, y, c, mk, ms, label=_beta_tex(beta))
    _plot_analytic(ax, xd, Fnorm_const, "black", label=rf"const $m$: $(r_{{ref}}/r)^4$, $r_{{ref}}={R_REF:g}$")
    ax.set_xlabel(xlab, fontsize=AXES_FONTSIZE, labelpad=12)
    ax.set_ylabel(r"$F/F_{\mathrm{ref}}$", fontsize=AXES_FONTSIZE, labelpad=16)

    ax.legend(frameon=False, prop={"size": LEGEND_FONTSIZE}, loc="best")
    _savefig(fig, ax, OUTDIR / "plot_side_force_norm_2R.png")

    # 4) Normalized side energy
    fig, ax = _new_fig_ax()
    for beta in beta_targets:
        rows = sel(beta, "side")
        x = np.array([rr["r"] for rr in rows], dtype=np.float64)
        y = np.array([rr["U_norm_2R"] for rr in rows], dtype=np.float64)
        c, mk, ms = _beta_style(beta)
        _plot_data_series(ax, x, y, c, mk, ms, label=_beta_tex(beta))
    _plot_analytic(ax, xd, Unorm_const, "black", label=rf"const $m$: $(r_{{ref}}/r)^3$, $r_{{ref}}={R_REF:g}$")
    ax.set_xlabel(xlab, fontsize=AXES_FONTSIZE, labelpad=12)
    ax.set_ylabel(r"$U/U_{\mathrm{ref}}$", fontsize=AXES_FONTSIZE, labelpad=16)

    ax.legend(frameon=False, prop={"size": LEGEND_FONTSIZE}, loc="best")
    _savefig(fig, ax, OUTDIR / "plot_side_energy_norm_2R.png")

    # 5) Raw magnitude head force |F| (log-log)
    fig, ax = _new_fig_ax()
    for beta in beta_targets:
        rows = sel(beta, "head")
        x = np.array([rr["r"] for rr in rows], dtype=np.float64)
        Fm = np.array([rr["F_meas"] for rr in rows], dtype=np.float64)
        Fa = np.array([rr["F_point_single"] for rr in rows], dtype=np.float64)
        c, mk, ms = _beta_style(beta)

        ax.loglog(
            x, np.abs(Fm),
            color=c, linewidth=PLOT_LW,
            marker=mk, markersize=ms,
            markeredgecolor=c, markerfacecolor="white", markeredgewidth=PLOT_LW,
            linestyle=":",
            zorder=ZORDER_DATA,
            label=rf"HOOMD {_beta_tex(beta)}",
        )
        ax.loglog(
            x, np.abs(Fa),
            color=c, linewidth=PLOT_LW,
            linestyle="-",
            zorder=ZORDER_ANALYTIC,
            label=rf"const $m$ {_beta_tex(beta)}",
        )
    ax.set_xlabel(xlab, fontsize=AXES_FONTSIZE, labelpad=12)
    ax.set_ylabel(r"$|F|$", fontsize=AXES_FONTSIZE, labelpad=16)

    ax.legend(frameon=False, prop={"size": LEGEND_FONTSIZE}, loc="best")
    _savefig(fig, ax, OUTDIR / "plot_head_force_raw_mag.png")

    # 6) Raw magnitude head energy |U| (log-log)
    fig, ax = _new_fig_ax()
    for beta in beta_targets:
        rows = sel(beta, "head")
        x = np.array([rr["r"] for rr in rows], dtype=np.float64)
        Um = np.array([rr["U_meas"] for rr in rows], dtype=np.float64)
        Ua = np.array([rr["U_point_single"] for rr in rows], dtype=np.float64)
        c, mk, ms = _beta_style(beta)

        ax.loglog(
            x, np.abs(Um),
            color=c, linewidth=PLOT_LW,
            marker=mk, markersize=ms,
            markeredgecolor=c, markerfacecolor="white", markeredgewidth=PLOT_LW,
            linestyle=":",
            zorder=ZORDER_DATA,
            label=rf"HOOMD {_beta_tex(beta)}",
        )
        ax.loglog(
            x, np.abs(Ua),
            color=c, linewidth=PLOT_LW,
            linestyle="-",
            zorder=ZORDER_ANALYTIC,
            label=rf"const $m$ {_beta_tex(beta)}",
        )
    ax.set_xlabel(xlab, fontsize=AXES_FONTSIZE, labelpad=12)
    ax.set_ylabel(r"$|U|$", fontsize=AXES_FONTSIZE, labelpad=16)

    ax.legend(frameon=False, prop={"size": LEGEND_FONTSIZE}, loc="best")
    _savefig(fig, ax, OUTDIR / "plot_head_energy_raw_mag.png")

    # 7) Raw magnitude side force |F| (log-log)
    fig, ax = _new_fig_ax()
    for beta in beta_targets:
        rows = sel(beta, "side")
        x = np.array([rr["r"] for rr in rows], dtype=np.float64)
        Fm = np.array([rr["F_meas"] for rr in rows], dtype=np.float64)
        Fa = np.array([rr["F_point_single"] for rr in rows], dtype=np.float64)
        c, mk, ms = _beta_style(beta)

        ax.loglog(
            x, np.abs(Fm),
            color=c, linewidth=PLOT_LW,
            marker=mk, markersize=ms,
            markeredgecolor=c, markerfacecolor="white", markeredgewidth=PLOT_LW,
            linestyle=":",
            zorder=ZORDER_DATA,
            label=rf"HOOMD {_beta_tex(beta)}",
        )
        ax.loglog(
            x, np.abs(Fa),
            color=c, linewidth=PLOT_LW,
            linestyle="-",
            zorder=ZORDER_ANALYTIC,
            label=rf"const $m$ {_beta_tex(beta)}",
        )
    ax.set_xlabel(xlab, fontsize=AXES_FONTSIZE, labelpad=12)
    ax.set_ylabel(r"$|F|$", fontsize=AXES_FONTSIZE, labelpad=16)

    ax.legend(frameon=False, prop={"size": LEGEND_FONTSIZE}, loc="best")
    _savefig(fig, ax, OUTDIR / "plot_side_force_raw_mag.png")

    # 8) Raw magnitude side energy |U| (log-log)
    fig, ax = _new_fig_ax()
    for beta in beta_targets:
        rows = sel(beta, "side")
        x = np.array([rr["r"] for rr in rows], dtype=np.float64)
        Um = np.array([rr["U_meas"] for rr in rows], dtype=np.float64)
        Ua = np.array([rr["U_point_single"] for rr in rows], dtype=np.float64)
        c, mk, ms = _beta_style(beta)

        ax.loglog(
            x, np.abs(Um),
            color=c, linewidth=PLOT_LW,
            marker=mk, markersize=ms,
            markeredgecolor=c, markerfacecolor="white", markeredgewidth=PLOT_LW,
            linestyle=":",
            zorder=ZORDER_DATA,
            label=rf"HOOMD {_beta_tex(beta)}",
        )
        ax.loglog(
            x, np.abs(Ua),
            color=c, linewidth=PLOT_LW,
            linestyle="-",
            zorder=ZORDER_ANALYTIC,
            label=rf"const $m$ {_beta_tex(beta)}",
        )
    ax.set_xlabel(xlab, fontsize=AXES_FONTSIZE, labelpad=12)
    ax.set_ylabel(r"$|U|$", fontsize=AXES_FONTSIZE, labelpad=16)

    ax.legend(frameon=False, prop={"size": LEGEND_FONTSIZE}, loc="best")
    _savefig(fig, ax, OUTDIR / "plot_side_energy_raw_mag.png")

    # 9) Ratio plots vs pair-analytic
    fig, ax = _new_fig_ax()
    for beta in beta_targets:
        c, _, _ = _beta_style(beta)

        rows_h = sel(beta, "head")
        xh = np.array([rr["r"] for rr in rows_h], dtype=np.float64)
        yh = np.array([rr["ratio_pairF"] for rr in rows_h], dtype=np.float64)
        _plot_data_series(ax, xh, yh, c, "o", MARKER_SIZE, label=rf"head {_beta_tex(beta)}")

        rows_s = sel(beta, "side")
        xs = np.array([rr["r"] for rr in rows_s], dtype=np.float64)
        ys = np.array([rr["ratio_pairF"] for rr in rows_s], dtype=np.float64)
        _plot_data_series(ax, xs, ys, c, "s", MARKER_SIZE - 1, label=rf"side {_beta_tex(beta)}")

    ax.axhline(1.0, color="black", linewidth=PLOT_LW, linestyle="-", zorder=ZORDER_REF)
    ax.set_xlabel(xlab, fontsize=AXES_FONTSIZE, labelpad=12)
    ax.set_ylabel(r"$F/F_{\mathrm{pair}}$", fontsize=AXES_FONTSIZE, labelpad=16)

    ax.legend(frameon=False, prop={"size": LEGEND_FONTSIZE}, loc="best")
    _savefig(fig, ax, OUTDIR / "plot_ratio_pairF.png")

    fig, ax = _new_fig_ax()
    for beta in beta_targets:
        c, _, _ = _beta_style(beta)

        rows_h = sel(beta, "head")
        xh = np.array([rr["r"] for rr in rows_h], dtype=np.float64)
        yh = np.array([rr["ratio_pairU"] for rr in rows_h], dtype=np.float64)
        _plot_data_series(ax, xh, yh, c, "o", MARKER_SIZE, label=rf"head {_beta_tex(beta)}")

        rows_s = sel(beta, "side")
        xs = np.array([rr["r"] for rr in rows_s], dtype=np.float64)
        ys = np.array([rr["ratio_pairU"] for rr in rows_s], dtype=np.float64)
        _plot_data_series(ax, xs, ys, c, "s", MARKER_SIZE - 1, label=rf"side {_beta_tex(beta)}")

    ax.axhline(1.0, color="black", linewidth=PLOT_LW, linestyle="-", zorder=ZORDER_REF)
    ax.set_xlabel(xlab, fontsize=AXES_FONTSIZE, labelpad=12)
    ax.set_ylabel(r"$U/U_{\mathrm{pair}}$", fontsize=AXES_FONTSIZE, labelpad=16)

    ax.legend(frameon=False, prop={"size": LEGEND_FONTSIZE}, loc="best")
    _savefig(fig, ax, OUTDIR / "plot_ratio_pairU.png")

    # 10) Dipole enhancement plot (m_pair / m_single)
    fig, ax = _new_fig_ax()
    for beta in beta_targets:
        c, _, _ = _beta_style(beta)

        rows_h = sel(beta, "head")
        xh = np.array([rr["r"] for rr in rows_h], dtype=np.float64)
        yh = np.array([rr["enh_pair_over_single"] for rr in rows_h], dtype=np.float64)
        _plot_data_series(ax, xh, yh, c, "o", MARKER_SIZE, label=rf"head {_beta_tex(beta)}")

        rows_s = sel(beta, "side")
        xs = np.array([rr["r"] for rr in rows_s], dtype=np.float64)
        ys = np.array([rr["enh_pair_over_single"] for rr in rows_s], dtype=np.float64)
        _plot_data_series(ax, xs, ys, c, "s", MARKER_SIZE - 1, label=rf"side {_beta_tex(beta)}")

    ax.axhline(1.0, color="black", linewidth=PLOT_LW, linestyle="-", zorder=ZORDER_REF)
    ax.set_xlabel(xlab, fontsize=AXES_FONTSIZE, labelpad=12)
    ax.set_ylabel(r"$m_{\mathrm{pair}}/m_{\mathrm{single}}$", fontsize=AXES_FONTSIZE, labelpad=16)

    ax.legend(frameon=False, prop={"size": LEGEND_FONTSIZE}, loc="best")
    _savefig(fig, ax, OUTDIR / "plot_m_enhancement.png")


# ----------------------------
# Additional HOOMD-internal kernel tests (fixed dipoles)
# ----------------------------


def fixed_pair_eval(r: float, mode: str, mu: float, pref: float) -> Tuple[float, float, int]:
    r = float(r)
    if mode == "head":
        pos = np.array([[0.0, 0.0, -0.5 * r],
                        [0.0, 0.0, +0.5 * r]], dtype=np.float64)
        comp = 2
    elif mode == "side":
        pos = np.array([[-0.5 * r, 0.0, 0.0],
                        [+0.5 * r, 0.0, 0.0]], dtype=np.float64)
        comp = 0
    else:
        raise ValueError("mode must be head/side")

    force = FixedDipoleForce(Hhat=HHAT, mu=mu, r_cut=R_CUT, prefactor=pref,
                             periodic_z=DIPOLE_PERIODIC_Z, use_cell_list=DIPOLE_USE_CELL_LIST)
    ev = StaticEvaluator(N=2, force_obj=force, seed=33)
    _, f, e = ev.eval(pos)

    f_comp = float(f[1, comp])
    U_sum = float(np.sum(e))
    return f_comp, U_sum, int(force.calls)


def run_additional_tests():
    print("\nADDITIONAL TESTS (fixed dipoles inside HOOMD)\n", flush=True)

    Fh, Uh, calls = fixed_pair_eval(r=2.0, mode="head", mu=1.0, pref=1.0)
    print(f"[T1] head sign: Fz_top={Fh:.6g} (expect <0) calls={calls} -> {'PASS' if (Fh < 0.0) else 'FAIL'}", flush=True)

    Fs, Us, calls2 = fixed_pair_eval(r=2.0, mode="side", mu=1.0, pref=1.0)
    print(f"[T2] side sign: Fx_right={Fs:.6g} (expect >0) calls={calls2} -> {'PASS' if (Fs > 0.0) else 'FAIL'}", flush=True)

    F1, _, _ = fixed_pair_eval(r=2.0, mode="head", mu=1.0, pref=1.0)
    F3, _, _ = fixed_pair_eval(r=2.0, mode="head", mu=1.0, pref=3.0)
    val3 = (F3 / F1) if abs(F1) > 0 else float("nan")
    ok3 = (abs(val3 - 3.0) < 1e-12)
    print(f"[T3] pref scaling: F(pref=3)/F(pref=1)={val3:.6g} (expect 3) -> {'PASS' if ok3 else 'FAIL'}", flush=True)

    Fm1, _, _ = fixed_pair_eval(r=2.0, mode="head", mu=1.0, pref=1.0)
    Fm2, _, _ = fixed_pair_eval(r=2.0, mode="head", mu=2.0, pref=1.0)
    val4 = (Fm2 / Fm1) if abs(Fm1) > 0 else float("nan")
    ok4 = (abs(val4 - 4.0) < 1e-12)
    print(f"[T4] mu^2 scaling: F(mu=2)/F(mu=1)={val4:.6g} (expect 4) -> {'PASS' if ok4 else 'FAIL'}", flush=True)

    r0 = 2.0
    dr = float(FD_DR)
    F0, U0, _ = fixed_pair_eval(r=r0, mode="head", mu=1.0, pref=1.0)
    _, U_plus, _ = fixed_pair_eval(r=r0 + dr, mode="head", mu=1.0, pref=1.0)
    _, U_minus, _ = fixed_pair_eval(r=r0 - dr, mode="head", mu=1.0, pref=1.0)
    dUdr = (U_plus - U_minus) / (2.0 * dr)
    ref = -dUdr
    rel = abs((F0 - ref) / (ref + 1e-300))
    ok5 = (rel < 2e-3)
    print(f"[T5] FD check (head): F={F0:.6g} ref=-dU/dr={ref:.6g} rel_err={rel:.3g} -> {'PASS' if ok5 else 'FAIL'}", flush=True)

    print(
        "\nNotes\n"
        "  These tests isolate dipole_forces_ext under fixed dipoles.\n"
        "  If they fail, the kernel or the CustomForce write path is wrong.\n",
        flush=True,
    )


# ----------------------------
# Main
# ----------------------------


def main():
    records: List[Dict] = []
    ratio_pairF: List[float] = []
    ratio_pairU: List[float] = []

    print("\nSERIES (mutual polarization inside HOOMD)\n", flush=True)

    for beta_target in BETA_TARGETS:
        eps_p = eps_from_beta(beta_target)
        beta_eff = beta_eff_from_eps(eps_p)
        print(f"[beta={beta_target:g}] beta_eff={beta_eff:.6g} eps_p={eps_p:.6g}", flush=True)

        m_single = measure_single_dipole(eps_p)
        print(f"  m_single={m_single:.6g} (unit field, HOOMD CustomForce)", flush=True)

        for mode in ("head", "side"):
            series = measure_pair_series(eps_p, mode=mode)
            for meas in series:
                r_meas = float(meas["r"])
                F_meas = float(meas["f_comp"])
                U_meas = float(meas["U_sum"])
                m0 = float(meas["m0"])
                m1 = float(meas["m1"])
                m_pair = 0.5 * (m0 + m1)

                F_single = point_dipole_F(PREF, m_single, r_meas, mode)
                U_single = point_dipole_U(PREF, m_single, r_meas, mode)

                F_pair = point_dipole_F(PREF, m_pair, r_meas, mode)
                U_pair = point_dipole_U(PREF, m_pair, r_meas, mode)

                F_ref_single = point_dipole_F(PREF, m_single, float(R_REF), mode)
                U_ref_single = point_dipole_U(PREF, m_single, float(R_REF), mode)

                F_norm_2R = F_meas / (F_ref_single + 1e-300)
                U_norm_2R = U_meas / (U_ref_single + 1e-300)

                F_norm_const = (float(R_REF) / float(r_meas)) ** 4
                U_norm_const = (float(R_REF) / float(r_meas)) ** 3

                rF = (F_meas / F_pair) if abs(F_pair) > 0 else float("nan")
                rU = (U_meas / U_pair) if abs(U_pair) > 0 else float("nan")

                ratio_pairF.append(rF)
                ratio_pairU.append(rU)

                records.append(dict(
                    beta_target=float(beta_target),
                    beta_eff=float(beta_eff),
                    eps_p=float(eps_p),
                    mode=mode,
                    r=float(r_meas),
                    r_ref_2R=float(R_REF),
                    m_single=float(m_single),
                    m0=float(m0),
                    m1=float(m1),
                    m_pair=float(m_pair),
                    enh_pair_over_single=float(m_pair / (m_single + 1e-300)),
                    F_meas=float(F_meas),
                    U_meas=float(U_meas),
                    F_point_single=float(F_single),
                    U_point_single=float(U_single),
                    F_point_pair=float(F_pair),
                    U_point_pair=float(U_pair),
                    ratio_pairF=float(rF),
                    ratio_pairU=float(rU),
                    F_ref_single_2R=float(F_ref_single),
                    U_ref_single_2R=float(U_ref_single),
                    F_norm_2R=float(F_norm_2R),
                    U_norm_2R=float(U_norm_2R),
                    F_norm_const_2R=float(F_norm_const),
                    U_norm_const_2R=float(U_norm_const),
                    calls=int(meas["calls"]),
                ))

            if mode == "head":
                rr2 = [rr for rr in records
                       if rr["beta_target"] == float(beta_target)
                       and rr["mode"] == "head"
                       and abs(rr["r"] - 2.0) < 1e-6]
                if rr2:
                    rr = rr2[-1]
                    print(
                        f"  [head r=2] F={rr['F_meas']:.6g} U={rr['U_meas']:.6g} "
                        f"ratioF(pair)={rr['ratio_pairF']:.6g} ratioU(pair)={rr['ratio_pairU']:.6g} "
                        f"F_norm_2R={rr['F_norm_2R']:.6g} U_norm_2R={rr['U_norm_2R']:.6g}",
                        flush=True,
                    )

    csv_path = OUTDIR / "mutual_polarization_force_energy_series.csv"
    if not records:
        raise RuntimeError("No records produced.")

    keys = list(records[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in records:
            w.writerow(row)
    print("\n[wrote]", csv_path, flush=True)

    make_plots(records, BETA_TARGETS)

    ratio_pairF_arr = np.asarray(ratio_pairF, dtype=np.float64)
    ratio_pairU_arr = np.asarray(ratio_pairU, dtype=np.float64)

    print("\n[sanity] ratio_pairF stats (expect ~1):", flush=True)
    print(
        f"  mean={float(np.nanmean(ratio_pairF_arr)):.6g} "
        f"p50={float(np.nanmedian(ratio_pairF_arr)):.6g} "
        f"min={float(np.nanmin(ratio_pairF_arr)):.6g} "
        f"max={float(np.nanmax(ratio_pairF_arr)):.6g}",
        flush=True,
    )

    print("[sanity] ratio_pairU stats (expect ~1):", flush=True)
    print(
        f"  mean={float(np.nanmean(ratio_pairU_arr)):.6g} "
        f"p50={float(np.nanmedian(ratio_pairU_arr)):.6g} "
        f"min={float(np.nanmin(ratio_pairU_arr)):.6g} "
        f"max={float(np.nanmax(ratio_pairU_arr)):.6g}",
        flush=True,
    )

    run_additional_tests()


if __name__ == "__main__":
    main()
