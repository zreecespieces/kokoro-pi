"""Where this package's own files are, whether it was cloned or installed.

Two layouts have to work and they are not the same shape:

    a checkout      kokoro-pi/native/…      kokoro-pi/src/kokoro_pi/
    an installed    …/site-packages/kokoro_pi/_bundled/native/…

`native/` and `corpus/` stay at the top of the repository because they are the
most interesting things in it, and a wheel gets a copy inside the package -- see
`setup.py`. Everything that needs one of those files asks here rather than
counting `..` and being wrong in one of the two layouts.
"""
from __future__ import annotations

import platform
from pathlib import Path

LIBRARY = "libkokoro_pi_ops.so"

#: Copied in at build time; absent in a checkout.
BUNDLED = Path(__file__).resolve().parent / "_bundled"
#: The repository, when this is running from one.
CHECKOUT = Path(__file__).resolve().parents[2]


def resource(relative: str) -> Path:
    """`native/build.sh`, `corpus/english.json`, and friends."""
    for root in (BUNDLED, CHECKOUT):
        candidate = root / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{relative} is missing. Looked in {BUNDLED} and {CHECKOUT}.\n"
        "  An installed kokoro-pi carries its own copy; a checkout keeps them at the top "
        "of the repository."
    )


#: The same library, compiled for a CPU with ARM's dot-product extension.
DOTPROD_LIBRARY = "libkokoro_pi_ops.dotprod.so"


def has_dotprod() -> bool:
    """Whether this CPU can run `SDOT`.

    The kernel picks its path at *compile* time -- `#if
    defined(__ARM_FEATURE_DOTPROD)` -- so a library built for a Pi 5 executes an
    illegal instruction on a Pi 4's Cortex-A72. A published wheel cannot know
    which it will land on, so it carries both and this decides.
    """
    if platform.machine() not in ("aarch64", "arm64"):
        return False
    if platform.system() == "Darwin":
        return True  # every Apple Silicon core has it
    try:
        return "asimddp" in Path("/proc/cpuinfo").read_text()
    except OSError:
        return False


def prebuilt_library() -> Path | None:
    """The compiled operators, when the wheel shipped them.

    A binary wheel carries libraries built in a manylinux container for this
    architecture, which is what lets `pip install` work on a machine with no
    compiler. A source install has none and builds its own.
    """
    if has_dotprod():
        fast = BUNDLED / DOTPROD_LIBRARY
        if fast.exists():
            return fast
    candidate = BUNDLED / LIBRARY
    return candidate if candidate.exists() else None
