#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""chain10_mutualpol_tests_wca_v6_pkg.py

HOOMD v6.x tests (N=10) using mutual polarization (capacitance) + dipole_forces_ext.

Packaging assumption
--------------------
The dipole code is installed as a single Python package:
  - dipole_forces_ext._dipole_forces_ext (compiled) : real-space dipole forces/energies
  - dipole_forces_ext.capacitance (pure Python)     : mutual polarization solver
  - dipole_forces_ext.ewald (pure Python)           : Spectral-Ewald utilities

If the compiled extension is not yet built in the active Python environment, this script
can optionally attempt an in-place build via setup_dipole_forces_ext.py.

Cases
-----
  1) chain_parallel_lowkT:  10-particle chain along field (z), low kT
  2) chain_parallel_highkT: 10-particle chain along field (z), high kT
  3) chain_perp_highkT:     10-particle chain perpendicular to field (x), high kT

Run length
----------
TOTAL_TIME is in reduced time units (tauD=1 in these reduced units).
"""

from __future__ import annotations

import sys
import math
import time
import subprocess
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import imageio.v2 as imageio
import gsd.hoomd

import hoomd
import hoomd.md


# ----------------------------
# Build/import dipole_forces_ext package
# ----------------------------

AUTO_BUILD_DIPOLE_FORCES_EXT = True


def _import_dipole_pkg():
    """Import dipole_forces_ext; optionally build in-place if missing."""
    try:
        import dipole_forces_ext  # noqa: F401
        from dipole_forces_ext import capacitance  # noqa: F401
        return dipole_forces_ext, capacitance
    except Exception as first_exc:
        if not AUTO_BUILD_DIPOLE_FORCES_EXT:
            raise

        here = Path(__file__).resolve().parent
        build_script = here / "setup_dipole_forces_ext.py"
        if not build_script.exists():
            raise RuntimeError(
                "dipole_forces_ext import failed and setup_dipole_forces_ext.py was not found next to this script. "
                "Either install the package, or place setup_dipole_forces_ext.py alongside this file."
            ) from first_exc

        cmd = [sys.executable, str(build_script), "build_ext", "--inplace"]
        print("[build] running:", " ".join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=str(here))

        import importlib

        dipole_forces_ext = importlib.import_module("dipole_forces_ext")
        capacitance = importlib.import_module("dipole_forces_ext.capacitance")
        return dipole_forces_ext, capacitance


dipole_forces_ext, capacitance = _import_dipole_pkg()


# ----------------------------
# I/O
# ----------------------------

OUTDIR = Path("out_chain_tests")
OUTDIR.mkdir(exist_ok=True)


# ----------------------------
# Global physical / numerical params
# ----------------------------

SEED = 42

# System
N = 10
BOX_L = 40.0
BOX = np.array([BOX_L, BOX_L, BOX_L], dtype=np.float64)

SIGMA = 1.0

# Dipole kernel (real-space cutoff)
R_CUT = 15.0
PREF = 2.5
HHAT = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # field along +z

# Mutual polarization (capacitance) parameters
XI = 0.5
ERRTOL = 1.0e-6
SLAB_FACTOR = 1.0
USE_SCIPY_GMRES = True

# Dipole-force kernel options
DIPOLE_PERIODIC_Z = True
DIPOLE_USE_CELL_LIST = True

# Polarizability control: beta -> eps_p
BETA_TARGET = 1.0  # "high polarizability"
MU_TARGET = 3.0    # target single-particle dipole moment under external field

# Brownian dynamics
GAMMA = 6.0
DT = 1.0e-4

# Run length
TOTAL_TIME = 0.5

# Output cadence
FRAMES_PER_TAU = 200
FRAME_DT = 1.0 / FRAMES_PER_TAU
FPS = 30

# Mutual polarization solve cadence (physical time). Bigger -> faster.
POL_UPDATE_DT = 0.01

# WCA repulsion (LJ truncated at r_cut = 2^(1/6)*sigma)
WCA_EPS = 80.0
WCA_R_CUT = float(2.0 ** (1.0 / 6.0) * SIGMA)

# Chain initialization
CHAIN_SPACING = 1.5
CHAIN_JITTER = 0.01


# ----------------------------
# beta <-> eps_p helpers
# ----------------------------

def eps_from_beta(beta_target: float) -> float:
    """Invert beta=(eps-1)/(eps+2). For beta->1, use large eps."""
    b = float(beta_target)
    if b >= 0.999999:
        return 1.0e6
    if b <= -0.999999:
        return 1.0e-6
    return (1.0 + 2.0 * b) / (1.0 - b)


def beta_eff_from_eps(eps_p: float) -> float:
    e = float(eps_p)
    return (e - 1.0) / (e + 2.0)


# ----------------------------
# Geometry helpers
# ----------------------------

def make_chain_positions(n: int, spacing: float, along: str, jitter: float, rng: np.random.Generator):
    """Create a straight chain centered near origin."""
    along = along.lower()
    coords = np.zeros((n, 3), dtype=np.float64)
    x = (np.arange(n, dtype=np.float64) - 0.5 * (n - 1)) * float(spacing)

    if along == "x":
        coords[:, 0] = x
    elif along == "y":
        coords[:, 1] = x
    elif along == "z":
        coords[:, 2] = x
    else:
        raise ValueError("along must be 'x','y','z'")

    if jitter > 0:
        coords += rng.normal(scale=float(jitter), size=coords.shape)
    return coords


def min_image(dr: np.ndarray, boxL: np.ndarray) -> np.ndarray:
    out = dr.copy()
    out -= boxL * np.round(out / boxL)
    return out


def pair_stats_by_tag(pos: np.ndarray, tags: np.ndarray, boxL: np.ndarray, sigma: float):
    """Compute min separation + overlap counts in tag order."""
    pos_by_tag = np.zeros((N, 3), dtype=np.float64)
    for i in range(pos.shape[0]):
        pos_by_tag[int(tags[i])] = pos[i]

    dr_ee = min_image(pos_by_tag[N - 1] - pos_by_tag[0], boxL)
    end_to_end = float(np.linalg.norm(dr_ee))

    min_r = float("inf")
    n_pairs = 0
    n_lt_sigma = 0
    n_lt_099 = 0

    for i in range(N - 1):
        for j in range(i + 1, N):
            n_pairs += 1
            dr = min_image(pos_by_tag[j] - pos_by_tag[i], boxL)
            r = float(np.linalg.norm(dr))
            if r < min_r:
                min_r = r
            if r < sigma:
                n_lt_sigma += 1
            if r < 0.99 * sigma:
                n_lt_099 += 1

    return min_r, end_to_end, n_pairs, n_lt_sigma, n_lt_099


# ----------------------------
# Mutual polarization dipole force with caching
# ----------------------------


# ----------------------------
# HOOMD CustomForce (from dipole_forces_ext package)
# ----------------------------

from dipole_forces_ext.hoomd_force import MutualPolarizationDipoleForceCached

# Measure m_single for unit field (to pick field_strength for mu_target)
# ----------------------------

def measure_m_single_unit_field(eps_p: float) -> float:
    """Return m_single (dipole projection along HHAT) for one particle under unit field_strength."""
    snap = hoomd.Snapshot()
    snap.particles.N = 1
    snap.particles.types = ["A"]
    snap.particles.typeid[:] = 0
    snap.particles.position[:] = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    snap.configuration.box = [float(BOX[0]), float(BOX[1]), float(BOX[2]), 0, 0, 0]

    dev = hoomd.device.CPU()
    sim = hoomd.Simulation(device=dev, seed=123)
    sim.create_state_from_snapshot(snap)

    dip_force = MutualPolarizationDipoleForceCached(
        eps_p=float(eps_p),
        xi=XI,
        errtol=ERRTOL,
        slab_factor=SLAB_FACTOR,
        Hhat=HHAT,
        r_cut=R_CUT,
        prefactor=PREF,
        field_strength=1.0,
        pol_every_steps=1,
        use_scipy_gmres=USE_SCIPY_GMRES,
        periodic_z=DIPOLE_PERIODIC_Z,
        use_cell_list=DIPOLE_USE_CELL_LIST,
    )

    brown = hoomd.md.methods.Brownian(filter=hoomd.filter.All(), kT=0.0)
    brown.gamma["A"] = 1.0e12  # freeze

    integrator = hoomd.md.Integrator(dt=1.0e-12, methods=[brown], forces=[dip_force])
    integrator.integrate_rotational_dof = False
    sim.operations.integrator = integrator

    sim.run(1)

    if dip_force.last_dipoles is None:
        raise RuntimeError("Failed to compute m_single: dipoles not available.")
    return float(np.dot(dip_force.last_dipoles[0], HHAT))


# ----------------------------
# Rendering
# ----------------------------

def render_gsd_to_mp4_xz(
    gsd_path: Path,
    mp4_path: Path,
    dt: float,
    title: str,
    Lx: float,
    Lz: float,
    fps: int = 30,
    marker_size: float = 140.0,
):
    with gsd.hoomd.open(str(gsd_path), mode="r") as traj:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
        ax.set_xlim(-0.5 * Lx, 0.5 * Lx)
        ax.set_ylim(-0.5 * Lz, 0.5 * Lz)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_aspect("equal", "box")

        sc = ax.scatter([], [], s=marker_size, facecolors="none", edgecolors="k", linewidths=1.5)

        ax.annotate(
            "",
            xy=(0.45 * Lx, 0.35 * Lz),
            xytext=(0.45 * Lx, 0.15 * Lz),
            arrowprops=dict(arrowstyle="->", linewidth=2),
        )
        ax.text(0.46 * Lx, 0.36 * Lz, "H", fontsize=10)

        with imageio.get_writer(str(mp4_path), fps=int(fps)) as w:
            for frame in traj:
                pos = np.asarray(frame.particles.position, dtype=np.float64)
                sc.set_offsets(np.column_stack([pos[:, 0], pos[:, 2]]))

                step = int(frame.configuration.step)
                t = float(step) * float(dt)
                ax.set_title(f"{title}\nstep={step}  t={t:.3f}")

                fig.canvas.draw()
                rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
                w.append_data(rgba[:, :, :3])

        plt.close(fig)


# ----------------------------
# Run one case
# ----------------------------

def run_one_case(case_name: str, init_pos: np.ndarray, kT: float, eps_p: float, field_strength: float):
    print("\n" + "=" * 72)
    print("[case]", case_name, flush=True)
    print("[hoomd] version=", hoomd.version.version, flush=True)
    print("[dipole_forces_ext] version=", getattr(dipole_forces_ext, "__version__", "unknown"), flush=True)
    print(f"[box] L={BOX.tolist()}  r_cut(dip)={R_CUT:g} pref={PREF:g}", flush=True)
    print(f"[pol] beta={BETA_TARGET:g} eps_p={eps_p:.6g} beta_eff={beta_eff_from_eps(eps_p):.6g}", flush=True)
    print(f"[field] HHAT={HHAT.tolist()}  mu_target={MU_TARGET:g}  field_strength={field_strength:.6g}", flush=True)
    print(f"[dipole kernel] periodic_z={DIPOLE_PERIODIC_Z} use_cell_list={DIPOLE_USE_CELL_LIST}", flush=True)

    total_steps = int(round(TOTAL_TIME / DT))
    gsd_every = max(1, int(round(FRAME_DT / DT)))
    frames_est = total_steps // gsd_every + 1

    pol_every = max(1, int(math.ceil(POL_UPDATE_DT / DT)))

    F_wca_at_sigma = 24.0 * float(WCA_EPS) / float(SIGMA)
    drift_wca_sigma = DT * F_wca_at_sigma / GAMMA

    print(f"[bd] kT={kT:g} gamma={GAMMA:g} dt={DT:g}  steps={total_steps}  time={TOTAL_TIME:g}", flush=True)
    print(f"[io] frames_per_tau={FRAMES_PER_TAU} frame_dt={FRAME_DT:g} -> gsd_every={gsd_every} (frames ~ {frames_est})", flush=True)
    print(f"[pol] POL_UPDATE_DT={POL_UPDATE_DT:g} -> pol_every={pol_every} steps", flush=True)
    print(f"[core] WCA: eps={WCA_EPS:g} sigma={SIGMA:g} r_cut={WCA_R_CUT:.6g}", flush=True)
    print(f"[core] WCA force at r=sigma: ~24*eps/sigma = {F_wca_at_sigma:.6g} -> drift/step ~ {drift_wca_sigma:.6g}", flush=True)

    snap = hoomd.Snapshot()
    snap.particles.N = int(init_pos.shape[0])
    snap.particles.types = ["A"]
    snap.particles.typeid[:] = 0
    snap.particles.position[:] = init_pos.astype(np.float32)
    snap.configuration.box = [float(BOX[0]), float(BOX[1]), float(BOX[2]), 0, 0, 0]

    dev = hoomd.device.CPU()
    sim = hoomd.Simulation(device=dev, seed=SEED)
    sim.create_state_from_snapshot(snap)

    nl = hoomd.md.nlist.Cell(buffer=0.4)
    wca = hoomd.md.pair.LJ(nlist=nl)
    wca.params[("A", "A")] = dict(epsilon=float(WCA_EPS), sigma=float(SIGMA))
    wca.r_cut[("A", "A")] = float(WCA_R_CUT)
    try:
        wca.mode = "shift"
    except Exception:
        pass

    dip_force = MutualPolarizationDipoleForceCached(
        eps_p=float(eps_p),
        xi=XI,
        errtol=ERRTOL,
        slab_factor=SLAB_FACTOR,
        Hhat=HHAT,
        r_cut=R_CUT,
        prefactor=PREF,
        field_strength=float(field_strength),
        pol_every_steps=int(pol_every),
        use_scipy_gmres=USE_SCIPY_GMRES,
        periodic_z=DIPOLE_PERIODIC_Z,
        use_cell_list=DIPOLE_USE_CELL_LIST,
    )

    brown = hoomd.md.methods.Brownian(filter=hoomd.filter.All(), kT=float(kT))
    brown.gamma["A"] = float(GAMMA)

    integrator = hoomd.md.Integrator(dt=float(DT), methods=[brown], forces=[wca, dip_force])
    integrator.integrate_rotational_dof = False
    sim.operations.integrator = integrator

    gsd_path = OUTDIR / f"{case_name}.gsd"
    mp4_path = OUTDIR / f"{case_name}.mp4"
    if gsd_path.exists():
        gsd_path.unlink()

    gsd_writer = hoomd.write.GSD(
        filename=str(gsd_path),
        trigger=hoomd.trigger.Periodic(int(gsd_every)),
        mode="wb",
        filter=hoomd.filter.All(),
        dynamic=["property"],
    )
    sim.operations.writers.append(gsd_writer)

    diag_every = max(1, total_steps // 20)
    t0 = time.perf_counter()
    last_wall = t0
    last_step = 0

    for _ in range(0, total_steps, diag_every):
        sim.run(int(min(diag_every, total_steps - int(sim.timestep))))

        step = int(sim.timestep)
        now = time.perf_counter()
        dstep = step - last_step
        dt_wall = now - last_wall
        tps = (dstep / dt_wall) if dt_wall > 0 else float("nan")

        with sim.state.cpu_local_snapshot as s:
            pos = np.asarray(s.particles.position, dtype=np.float64)
            tags = np.asarray(s.particles.tag, dtype=np.int64)
            netF = np.asarray(s.particles.net_force, dtype=np.float64)

        Fdip = dip_force.last_forces if dip_force.last_forces is not None else np.zeros_like(netF)
        Fwca = netF - Fdip

        Fdip_mag = np.linalg.norm(Fdip, axis=1)
        Fwca_mag = np.linalg.norm(Fwca, axis=1)

        Fdip_p99 = float(np.percentile(Fdip_mag, 99.0))
        Fwca_p99 = float(np.percentile(Fwca_mag, 99.0))

        Udip_sum = float(np.sum(dip_force.last_energy)) if dip_force.last_energy is not None else 0.0

        min_r, end_to_end, n_pairs, n_lt_sigma, n_lt_099 = pair_stats_by_tag(pos, tags, BOX, SIGMA)
        overlap_pct = max(0.0, (1.0 - min_r / SIGMA)) * 100.0

        prog = step / float(total_steps)
        elapsed = now - t0
        eta = (elapsed * (1.0 / prog - 1.0)) if prog > 1e-12 else float("inf")

        print(
            f"[diag] step={step:6d}/{total_steps} "
            f"Fdip_p99={Fdip_p99:10.4g} Udip_sum={Udip_sum:10.3g} "
            f"pol_solves={dip_force.pol_solves:6d} "
            f"Fwca_p99={Fwca_p99:10.4g} "
            f"min_r={min_r:7.4f} overlap={overlap_pct:6.2f}% "
            f"pairs<1.00σ={n_lt_sigma:2d}/{n_pairs} pairs<0.99σ={n_lt_099:2d}/{n_pairs} "
            f"end_to_end={end_to_end:7.3f} "
            f"tps={tps:7.2f} prog={100*prog:6.2f}% eta={eta/60:6.2f}min",
            flush=True,
        )

        last_wall = now
        last_step = step

    print(f"[run] wrote: {gsd_path}", flush=True)

    render_gsd_to_mp4_xz(
        gsd_path=gsd_path,
        mp4_path=mp4_path,
        dt=float(DT),
        title=case_name,
        Lx=float(BOX[0]),
        Lz=float(BOX[2]),
        fps=int(FPS),
    )
    print(f"[render] wrote: {mp4_path}", flush=True)


# ----------------------------
# Main
# ----------------------------

def main():
    print("HOOMD version:", hoomd.version.version, flush=True)
    print("[diag] dipole_forces_ext version:", getattr(dipole_forces_ext, "__version__", "unknown"), flush=True)

    eps_p = eps_from_beta(BETA_TARGET)
    beta_eff = beta_eff_from_eps(eps_p)
    print(f"[setup] beta={BETA_TARGET:g} -> eps_p={eps_p:.6g} beta_eff={beta_eff:.6g}", flush=True)

    m_single = measure_m_single_unit_field(eps_p)
    if abs(m_single) < 1e-12:
        raise RuntimeError("m_single(unit field) ~ 0. Mutual polarization solve likely failed.")
    field_strength = float(MU_TARGET) / float(m_single)
    print(f"[setup] m_single(unit field)={m_single:.6g} -> field_strength={field_strength:.6g} to target mu={MU_TARGET:g}", flush=True)

    total_steps = int(round(TOTAL_TIME / DT))
    gsd_every = max(1, int(round(FRAME_DT / DT)))
    frames_est = total_steps // gsd_every + 1
    pol_every = max(1, int(math.ceil(POL_UPDATE_DT / DT)))

    print(f"[setup] total_time={TOTAL_TIME:g}  dt={DT:g} -> total_steps={total_steps}", flush=True)
    print(f"[setup] frames_per_tau={FRAMES_PER_TAU} frame_dt={FRAME_DT:g} -> frames ~ {frames_est}", flush=True)
    print(f"[setup] POL_UPDATE_DT={POL_UPDATE_DT:g} -> pol_every={pol_every} steps", flush=True)
    print(f"[setup] WCA: eps={WCA_EPS:g} sigma={SIGMA:g} r_cut={WCA_R_CUT:.6g}", flush=True)

    rng = np.random.default_rng(SEED)

    pos1 = make_chain_positions(N, spacing=CHAIN_SPACING, along="z", jitter=CHAIN_JITTER, rng=rng)
    run_one_case("chain_parallel_lowkT", pos1, kT=0.001, eps_p=eps_p, field_strength=field_strength)

    pos2 = make_chain_positions(N, spacing=CHAIN_SPACING, along="z", jitter=CHAIN_JITTER, rng=rng)
    run_one_case("chain_parallel_highkT", pos2, kT=1.0, eps_p=eps_p, field_strength=field_strength)

    pos3 = make_chain_positions(N, spacing=CHAIN_SPACING, along="x", jitter=CHAIN_JITTER, rng=rng)
    run_one_case("chain_perp_highkT", pos3, kT=1.0, eps_p=eps_p, field_strength=field_strength)

    print("\n[done] outputs in:", OUTDIR.resolve(), flush=True)


if __name__ == "__main__":
    main()
