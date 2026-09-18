"""Fast streaming Kokoro text to speech for Raspberry Pi, on the CPU.

    from kokoro_pi import Kokoro
    kokoro = Kokoro()
    samples, rate = kokoro.create("Hello from my Raspberry Pi.", voice="af_heart")

`Kokoro` is kokoro-onnx's API over this project's kernels -- see `compat.py`.
There is also a service: `kokoro-pi serve`.
"""
__version__ = "1.2.0"

__all__ = ["Kokoro", "__version__"]


def __getattr__(name: str):
    # Imported on first use rather than at import time: reaching `Kokoro` loads
    # onnxruntime and numpy, which is several seconds a CLI should not spend
    # printing its own help.
    if name == "Kokoro":
        from .compat import Kokoro

        return Kokoro
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # Without this, `Kokoro` is invisible to tab completion and to dir(), because
    # it is not in the module dictionary until something asks for it.
    return sorted({*globals(), *__all__})
