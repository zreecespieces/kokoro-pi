"""Packaging, which has one job beyond the ordinary: carry the kernel.

The interesting part of this project is a C++ file compiled with ARM dot-product
instructions, and `pyproject.toml` alone cannot express "copy two directories
from the top of the repository into the package, and compile one of them".

So `build_py` does three things:

1. copies `native/` and `corpus/` into `kokoro_pi/_bundled/`, so an installed
   package can find its own build script and calibration texts (see paths.py);
2. compiles `libkokoro_pi_ops.so` into the same place, unless told not to --
   which is what makes `pip install` work on a machine with no compiler;
3. marks the wheel platform-specific, because a compiled library is in it.

Set `KOKORO_PI_NO_NATIVE=1` to skip the compile and produce a pure wheel that
builds its operators on first use. That is what the sdist path does.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.dist import Distribution

try:  # setuptools vendors it; the standalone package is the fallback
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # pragma: no cover
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

HERE = Path(__file__).resolve().parent
BUNDLE = ("native", "corpus")
LIBRARY = "libkokoro_pi_ops.so"
DOTPROD_LIBRARY = "libkokoro_pi_ops.dotprod.so"


class BuildPy(_build_py):
    def run(self) -> None:
        super().run()
        target = Path(self.build_lib) / "kokoro_pi" / "_bundled"
        target.mkdir(parents=True, exist_ok=True)
        for name in BUNDLE:
            source = HERE / name
            if not source.is_dir():
                continue
            shutil.copytree(source, target / name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*.so", "__pycache__"))
        if os.environ.get("KOKORO_PI_NO_NATIVE"):
            print("kokoro-pi: skipping the native build (KOKORO_PI_NO_NATIVE)")
            return
        script = target / "native" / "build.sh"
        if not script.exists():
            print("kokoro-pi: no native sources to build", file=sys.stderr)
            return
        # A wheel is compiled on a machine that is not the one it will run on,
        # and the kernel chooses its SDOT path with #if at compile time -- so a
        # single ARM library is either illegal on a Pi 4 or slow on a Pi 5.
        # Build both and let paths.py pick at import time.
        builds = [(LIBRARY, None)]
        if os.uname().machine in ("aarch64", "arm64"):
            builds = [(LIBRARY, "-march=armv8-a"),
                      (DOTPROD_LIBRARY, "-march=armv8.2-a+dotprod")]
        for name, arch in builds:
            self.compile_library(script, target / name, arch)

    def compile_library(self, script: Path, out: Path, arch: str | None) -> None:
        environment = dict(os.environ)
        if arch:
            environment["ARCH_FLAGS"] = arch
        try:
            # Absolute, because build.sh cd's to its own directory first and a
            # relative output path would then point somewhere else entirely.
            subprocess.run(["bash", str(script.resolve()), str(out.resolve())],
                           check=True, env=environment)
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            if os.environ.get("KOKORO_PI_REQUIRE_NATIVE"):
                # Releasing a wheel that is tagged for a platform and carries no
                # library for it is worse than releasing nothing.
                raise
            # Otherwise a source install on an unusual machine is still useful:
            # `kokoro-pi build` compiles the operators there.
            print(f"kokoro-pi: could not compile the operators here ({error}).\n"
                  f"           `kokoro-pi build` will compile them on the target machine.",
                  file=sys.stderr)


class BdistWheel(_bdist_wheel):
    """Platform-specific, but not Python-specific.

    The compiled library is loaded by ONNX Runtime, not imported by CPython, so
    it has no ABI tie to the interpreter -- one `py3-none-linux_aarch64` wheel
    serves every supported Python instead of one per minor version.
    """

    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = bool(os.environ.get("KOKORO_PI_NO_NATIVE"))

    def get_tag(self):
        _, _, platform = super().get_tag()
        return "py3", "none", platform


class BinaryDistribution(Distribution):
    """A wheel holding a compiled library is not portable, and must say so."""

    def has_ext_modules(self) -> bool:  # noqa: D102
        return not os.environ.get("KOKORO_PI_NO_NATIVE")


setup(cmdclass={"build_py": BuildPy, "bdist_wheel": BdistWheel},
      distclass=BinaryDistribution)
