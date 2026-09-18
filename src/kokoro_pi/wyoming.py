"""The Wyoming protocol, which is how Home Assistant talks to a voice service.

Wyoming is a line of JSON, optionally followed by more JSON and then binary:

    { "type": "...", "data": { ... }, "data_length": N, "payload_length": M }\\n
    <N bytes of JSON>      (optional, merged over `data`)
    <M bytes of payload>   (optional)

A TTS service answers `describe` with `info`, and `synthesize` with
`audio-start`, some `audio-chunk`s and `audio-stop`. That is the whole protocol
surface this needs.

It also implements the *streaming* synthesise events -- `synthesize-start`,
`synthesize-chunk`, `synthesize-stop` -- and advertises
`supports_synthesize_streaming`. That is not a nicety here: this project exists
because speech that starts sooner feels faster, and a sentence at a time from an
assistant is exactly the case clause streaming was built for.

No dependency on the `wyoming` package. It would be one more thing to install on
a Pi for a protocol that is a hundred lines, and the spec is stable.
"""
from __future__ import annotations

import json
import socketserver
import threading
from typing import Any, BinaryIO

import numpy as np

from . import __version__

RATE = 24000
WIDTH = 2
CHANNELS = 1
#: The conventional Wyoming port for a text-to-speech service, and what Home
#: Assistant's integration offers by default.
DEFAULT_PORT = 10200
#: How much audio goes in one chunk. Small enough to start playing quickly,
#: large enough not to spend the whole time in write().
CHUNK_SAMPLES = 2048

ATTRIBUTION = {"name": "kokoro-pi", "url": "https://github.com/zreecespieces/kokoro-pi"}


def write_event(stream: BinaryIO, kind: str, data: dict | None = None,
                payload: bytes | None = None) -> None:
    header: dict[str, Any] = {"type": kind}
    if data is not None:
        header["data"] = data
    if payload is not None:
        header["payload_length"] = len(payload)
    stream.write(json.dumps(header, ensure_ascii=False).encode() + b"\n")
    if payload is not None:
        stream.write(payload)
    stream.flush()


def read_event(stream: BinaryIO) -> tuple[str, dict] | None:
    """One event, or None at end of stream."""
    line = stream.readline()
    if not line:
        return None
    try:
        header = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(header, dict) or "type" not in header:
        return None
    data = header.get("data") or {}
    extra = header.get("data_length")
    if extra:
        try:
            data = {**data, **json.loads(stream.read(int(extra)))}
        except (ValueError, json.JSONDecodeError):
            return None
    payload = header.get("payload_length")
    if payload:
        stream.read(int(payload))  # a TTS service is never sent audio
    return str(header["type"]), data


def info_event(engine, languages_for) -> dict:
    """What this service is and which voices it has, in Wyoming's shape."""
    voices = [
        {
            "name": name,
            "attribution": {"name": "hexgrad/Kokoro-82M",
                            "url": "https://huggingface.co/hexgrad/Kokoro-82M"},
            "installed": True,
            "description": None,
            "version": "1.0",
            "languages": [languages_for(name)],
            "speakers": None,
        }
        for name in engine.voices()
    ]
    return {
        "asr": [], "handle": [], "intent": [], "wake": [], "mic": [], "snd": [],
        "tts": [{
            "name": "kokoro-pi",
            "attribution": ATTRIBUTION,
            "installed": True,
            "description": f"Kokoro-82M on the CPU, {engine.variant} build",
            "version": __version__,
            "voices": voices,
            "supports_synthesize_streaming": True,
        }],
    }


class Handler(socketserver.StreamRequestHandler):
    """One Home Assistant connection. Long-lived: it describes, then synthesises."""

    engine: Any = None
    settings: Any = None
    split: Any = None

    def handle(self) -> None:
        # Text arrives a chunk at a time in the streaming flow, so a connection
        # accumulates until it is told the sentence is over.
        buffered: list[str] = []
        streaming = False
        voice: str | None = None
        while True:
            event = read_event(self.rfile)
            if event is None:
                return
            kind, data = event
            try:
                if kind == "describe":
                    write_event(self.wfile, "info", info_event(self.engine, self.language_of))
                elif kind == "synthesize":
                    self.say(str(data.get("text") or ""), self.voice_of(data))
                elif kind == "synthesize-start":
                    buffered, streaming, voice = [], True, self.voice_of(data)
                elif kind == "synthesize-chunk" and streaming:
                    buffered.append(str(data.get("text") or ""))
                elif kind == "synthesize-stop" and streaming:
                    self.say("".join(buffered), voice)
                    write_event(self.wfile, "synthesize-stopped")
                    buffered, streaming, voice = [], False, None
            except (BrokenPipeError, ConnectionError):
                return
            except Exception:
                # A failed synthesis must not take the connection down with it:
                # Home Assistant keeps one open and would stop speaking for good.
                try:
                    write_event(self.wfile, "audio-stop")
                except (BrokenPipeError, ConnectionError):
                    return

    def language_of(self, name: str) -> str:
        return self.engine.language_for(name)

    def voice_of(self, data: dict) -> str | None:
        voice = data.get("voice") or {}
        name = voice.get("name") if isinstance(voice, dict) else None
        return name if name in self.engine.voices() else None

    def say(self, text: str, voice: str | None) -> None:
        text = text.strip()
        if not text:
            return
        audio_format = {"rate": RATE, "width": WIDTH, "channels": CHANNELS}
        # One request at a time, same as the HTTP side -- but a blocking wait
        # rather than a 429, because Wyoming has no "come back later" and Home
        # Assistant would simply lose the announcement.
        with self.engine.lock:
            write_event(self.wfile, "audio-start", audio_format)
            for piece in self.split(text, self.settings.stream_limit):
                audio = self.engine.synthesise(piece, voice)
                samples = np.clip(np.asarray(audio, np.float32) * 32767, -32768, 32767).astype("<i2")
                raw = samples.tobytes()
                step = CHUNK_SAMPLES * WIDTH
                for start in range(0, len(raw), step):
                    write_event(self.wfile, "audio-chunk", audio_format, raw[start:start + step])
            write_event(self.wfile, "audio-stop")


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(engine, settings, split, host: str, port: int) -> threading.Thread:
    """Start the Wyoming listener beside the HTTP one, sharing the engine."""
    handler = type("BoundHandler", (Handler,),
                   {"engine": engine, "settings": settings, "split": staticmethod(split)})
    server = Server((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, name="wyoming", daemon=True)
    thread.start()
    return thread
