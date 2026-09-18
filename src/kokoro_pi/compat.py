"""kokoro-onnx's API, backed by this project's kernels.

Anyone using Kokoro on a Pi today has written this:

    from kokoro_onnx import Kokoro
    kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    samples, rate = kokoro.create("Hello.", voice="af_heart")

and the only thing wrong with it is that it is slow. So this offers the same
object under the same method names:

    from kokoro_pi import Kokoro
    kokoro = Kokoro()
    samples, rate = kokoro.create("Hello.", voice="af_heart")

Two lines change and nothing else does -- `create`, `create_stream` and
`get_voices` take the arguments they always took and return what they always
returned.

The constructor is where the honesty is. It cannot take a model path, because
this project's models are *derived*: an int8 build with folded per-channel
scales and a native operator library beside it, made by `kokoro-pi build` and
described by a manifest. So it takes the directory holding that manifest, and
finds it the same way the service does -- the `models` setting, from a config
file, the environment, or the default `~/.kokoro-pi/models`.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator

if TYPE_CHECKING:  # numpy is only needed for the annotation
    import numpy as np


def default_models() -> Path:
    """Wherever this machine keeps its built models. Same answer as `kokoro-pi serve`."""
    from . import config as cfg

    settings = cfg.resolve(cfg.SERVE, None, "serve")
    if settings.models is not None:
        return Path(settings.models)
    return Path.home() / ".kokoro-pi" / "models"


class Kokoro:
    """A faster `kokoro_onnx.Kokoro`.

    Every method below is delegated to the real kokoro-onnx object, which is
    holding an ONNX Runtime session over the optimised graph with the native
    operators registered. So behaviour that is not about speed -- phonemisation,
    batching of long phoneme sequences, voice blending, trimming -- is not
    reimplemented here and cannot drift from upstream.
    """

    def __init__(self, models: str | Path | None = None, *, variant: str | None = None,
                 threads: int = 4, voice: str | None = None, lang: str = "auto",
                 verify: bool = True, warm: bool = True):
        from .server import Engine

        directory = Path(models) if models is not None else default_models()
        if not (directory / "models.json").exists():
            raise FileNotFoundError(
                f"no models.json in {directory}.\n"
                "  kokoro-pi derives its models rather than shipping them. Run:\n"
                f"    kokoro-pi build --models {directory}\n"
                "  (a 177 MB download and a few minutes), or pass models=<directory>."
            )
        self._engine = Engine(directory, variant, threads, voice, verify=verify,
                              lang=lang, warm=warm)
        #: The kokoro-onnx object underneath, if you need something not exposed here.
        self.kokoro = self._engine.kokoro
        self.models = directory

    # -- the kokoro-onnx surface ------------------------------------------

    def create(self, text: str, voice=None, speed: float = 1.0, lang: str | None = None,
               is_phonemes: bool = False, trim: bool = True) -> "tuple[np.ndarray, int]":
        """Synthesise, returning `(samples, sample_rate)` exactly as kokoro-onnx does.

        `voice` may be a name or a style vector, as upstream. `lang` defaults to
        the voice's own language rather than to American English -- `bf_emma`
        gets British phonemes -- and naming one explicitly still wins.
        """
        chosen = voice if voice is not None else self._engine.voice
        language = lang or (self._engine.language_for(chosen) if isinstance(chosen, str) else "en-us")
        return self.kokoro.create(text, voice=chosen, speed=speed, lang=language,
                                  is_phonemes=is_phonemes, trim=trim)

    def create_stream(self, text: str, voice=None, speed: float = 1.0, lang: str | None = None,
                      is_phonemes: bool = False,
                      trim: bool = True) -> "AsyncGenerator[tuple[np.ndarray, int], None]":
        """The upstream async generator, unchanged."""
        chosen = voice if voice is not None else self._engine.voice
        language = lang or (self._engine.language_for(chosen) if isinstance(chosen, str) else "en-us")
        return self.kokoro.create_stream(text, voice=chosen, speed=speed, lang=language,
                                         is_phonemes=is_phonemes, trim=trim)

    def get_voices(self) -> list[str]:
        return self._engine.voices()

    def get_voice_style(self, name: str):
        return self.kokoro.get_voice_style(name)

    # -- what this adds ----------------------------------------------------

    @property
    def variant(self) -> str:
        """`int8`, `float` or `upstream` -- which build is loaded."""
        return self._engine.variant

    @property
    def backend(self) -> str:
        return self._engine.backend

    def split_for_streaming(self, text: str, limit: int = 180) -> list[str]:
        """Clause-sized pieces, so a caller can stream without the HTTP service."""
        from .server import split_for_streaming

        return split_for_streaming(text, limit)

    def __repr__(self) -> str:
        return (f"<kokoro_pi.Kokoro variant={self.variant} backend={self.backend} "
                f"voices={len(self.get_voices())} models={self.models}>")
