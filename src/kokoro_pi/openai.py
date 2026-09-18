"""The OpenAI speech API, so existing clients need only a base URL.

`POST /v1/audio/speech` is what Open WebUI, LibreChat, SillyTavern and the
OpenAI SDKs already speak. Implementing it is worth more than any amount of
documentation about our own endpoint: it turns "port your client" into "change
one setting".

Two places where this has to bend, and both are answered in favour of the
client working rather than of being pedantic:

* **Compressed formats.** The spec's default is `mp3`, and nothing here can
  encode one. ffmpeg does it when ffmpeg is present; when it is not, the reply
  is a WAV with a header saying so, because a client that gets playable audio
  it did not ask for is in better shape than one that gets a 400.
* **Speed.** OpenAI allows 0.25-4.0 and Kokoro is sensible over 0.5-2.0, so
  values outside that are clamped rather than refused, with a header.
"""
from __future__ import annotations

import shutil
import subprocess

#: OpenAI's six voice names, mapped to the nearest Kokoro voice. Kokoro happens
#: to ship voices of the same name for five of them; `shimmer` has no
#: counterpart, so it goes to the closest American female.
OPENAI_VOICES = {
    "alloy": "af_alloy",
    "echo": "am_echo",
    "fable": "bm_fable",
    "onyx": "am_onyx",
    "nova": "af_nova",
    "shimmer": "af_sky",
    "ash": "am_adam",
    "sage": "af_sarah",
    "coral": "af_bella",
}

#: What ffmpeg is asked for, and what the response says it is.
ENCODERS = {
    "mp3": (["-f", "mp3", "-b:a", "64k"], "audio/mpeg"),
    "opus": (["-f", "ogg", "-c:a", "libopus", "-b:a", "48k"], "audio/ogg"),
    "aac": (["-f", "adts", "-c:a", "aac", "-b:a", "64k"], "audio/aac"),
    "flac": (["-f", "flac"], "audio/flac"),
}
#: `pcm` is handled by the caller, which has the samples before they become a
#: WAV -- it is deliberately not something `encode` will pretend to do.
NATIVE = {"wav": "audio/wav"}
FORMATS = (*NATIVE, "pcm", *ENCODERS)


def resolve_voice(requested: str | None, available: list[str], default: str) -> str | None:
    """An OpenAI voice name, a Kokoro voice name, or nothing. None means unknown."""
    if not requested:
        return default
    if requested in available:
        return requested
    mapped = OPENAI_VOICES.get(str(requested).lower())
    if mapped and mapped in available:
        return mapped
    # A service narrowed with `voices` may not hold the mapped voice. Falling
    # back to the default would silently ignore the request, so say instead.
    return None


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def encode(wav: bytes, fmt: str) -> tuple[bytes, str, bool]:
    """Transcode a WAV. Returns the bytes, the content type, and whether it fell back."""
    if fmt not in ENCODERS:
        return wav, "audio/wav", False
    if not have_ffmpeg():
        return wav, "audio/wav", True
    args, content_type = ENCODERS[fmt]
    try:
        done = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                               "-i", "pipe:0", *args, "pipe:1"],
                              input=wav, capture_output=True, timeout=120, check=True)
    except (subprocess.SubprocessError, OSError):
        return wav, "audio/wav", True
    return (done.stdout, content_type, False) if done.stdout else (wav, "audio/wav", True)


def clamp_speed(value: object) -> tuple[float, bool]:
    """OpenAI's range is wider than this model's. Clamp, and say that you did."""
    try:
        speed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1.0, True
    return (min(max(speed, 0.5), 2.0), not 0.5 <= speed <= 2.0)


def models(variant: str) -> dict:
    """`GET /v1/models`, which some clients call before they will show a voice list."""
    return {
        "object": "list",
        "data": [
            {"id": "kokoro", "object": "model", "owned_by": "kokoro-pi"},
            {"id": f"kokoro-{variant}", "object": "model", "owned_by": "kokoro-pi"},
            # Clients that hardcode OpenAI's ids should find something.
            {"id": "tts-1", "object": "model", "owned_by": "kokoro-pi"},
            {"id": "tts-1-hd", "object": "model", "owned_by": "kokoro-pi"},
        ],
    }
