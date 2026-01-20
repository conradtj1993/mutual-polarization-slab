from __future__ import annotations

import sys
from pathlib import Path

from setuptools import setup, find_packages

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except Exception as e:
    raise RuntimeError(
        "pybind11 is required to build this extension. Install with: pip install pybind11"
    ) from e

THIS_DIR = Path(__file__).resolve().parent
SRC = str(THIS_DIR / "dipole_forces_ext" / "_dipole_forces_ext.cpp")

compile_args = ["-O3", "-std=c++17"]
link_args: list[str] = []

# Enable OpenMP when available (Linux). macOS needs different flags/toolchain.
if sys.platform.startswith("linux"):
    compile_args += ["-fopenmp"]
    link_args += ["-fopenmp"]

ext_modules = [
    Pybind11Extension(
        "dipole_forces_ext._dipole_forces_ext",
        [SRC],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
]

setup(
    name="dipole_forces_ext",
    version="0.5.1",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
