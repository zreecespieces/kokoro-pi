"""Resident HTTP speech service.

Two things make this faster to *hear* than calling Kokoro directly:

* the optimised graph and native kernels synthesise about 2.5x quicker, and
* long text is split into clauses and each clause is streamed as soon as it is ready,
  so the first audio arrives after the first clause instead of after the whole passage.

The model stays resident, so there is no per-request load cost.
"""
from __future__ import annotations

import argparse
import hmac
import io
import json
import re
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import config as cfg
from . import openai as oai

RATE = 24000
FORMATS = set(cfg.FORMATS)
SENTENCE = re.compile(r"(?<=[.!?…])\s+|\n+")
LANG = re.compile(r"^[a-z]{2,3}(-[a-z]{2,4})?$")
CLAUSE = re.compile(r"(?<=[,;:])\s+")


def split_for_streaming(text: str, limit: int = 180) -> list[str]:
    """Break text into clause-sized pieces so the first audio can leave early.

    Kokoro already chunks internally, but it does so after receiving the whole
    request. Splitting here means each piece can be sent to the client while the next
    one is still being synthesised.
    """
    pieces: list[str] = []
    for sentence in (part.strip() for part in SENTENCE.split(text) if part.strip()):
        if len(sentence) <= limit:
            pieces.append(sentence)
            continue
        current = ""
        for clause in CLAUSE.split(sentence):
            candidate = f"{current} {clause}".strip()
            if current and len(candidate) > limit:
                pieces.append(current)
                current = clause
            else:
                current = candidate
        if current:
            # Still too long: fall back to word boundaries so nothing is unbounded.
            while len(current) > limit:
                cut = current.rfind(" ", 0, limit)
                cut = cut if cut > limit // 2 else limit
                pieces.append(current[:cut].strip())
                current = current[cut:].strip()
            if current:
                pieces.append(current)
    return pieces or [text.strip()]


def to_bytes(audio: np.ndarray, layout: str) -> bytes:
    clipped = np.clip(np.asarray(audio, np.float32) * 32767, -32768, 32767)
    return clipped.astype(">i2" if layout == "l16" else "<i2").tobytes()


def wav_bytes(audio: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(to_bytes(audio, "s16le"))
    return buffer.getvalue()


class Engine:
    """The resident model, plus whatever the manifest says about how it was built."""

    def __init__(self, models: Path, variant: str | None = None, threads: int = 4,
                 voice: str | None = None, verify: bool = True,
                 allowed: list[str] | None = None, lang: str = "auto", speed: float = 1.0,
                 warm: bool = True):
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        from .assets import digest

        self.models = models
        manifest_path = models / "models.json"
        if not manifest_path.exists():
            raise SystemExit(f"no models.json in {models}; run `kokoro-pi build` first")
        self.manifest = json.loads(manifest_path.read_text())
        self.variant = variant or self.manifest.get("variant", "int8")
        if self.variant not in self.manifest["variants"]:
            raise SystemExit(f"unknown variant {self.variant!r}; "
                             f"have {sorted(self.manifest['variants'])}")
        chosen = self.manifest["variants"][self.variant]
        self.backend = chosen["backend"]
        self.voice = voice or self.manifest.get("voice", "af_heart")
        self.lang = lang
        self.speed = speed
        self.allowed = [name for name in (allowed or []) if name]
        checksums = self.manifest.get("checksums", {})

        needed = [chosen["model"], "voices-v1.0.bin", *chosen.get("custom_ops", [])]
        for name in needed:
            path = models / name
            if not path.exists():
                raise SystemExit(f"{name} is missing from {models}")
            if verify and name in checksums and digest(path) != checksums[name]:
                raise SystemExit(f"checksum mismatch for {name}; rebuild or re-download")

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        for name in chosen.get("custom_ops", []):
            options.register_custom_ops_library(str(models / name))
        self.session = ort.InferenceSession(str(models / chosen["model"]), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.kokoro = Kokoro.from_session(self.session, str(models / "voices-v1.0.bin"))
        self.threads = threads
        self.lock = threading.Lock()
        available = self._pack_voices()
        if self.allowed:
            missing = [name for name in self.allowed if name not in available]
            if missing:
                raise SystemExit(f"voices not in the pack: {', '.join(missing)}\n"
                                 f"  have {len(available)}; see `kokoro-pi say --help` or /v1/voices")
            available = [name for name in available if name in self.allowed]
        self._voices = available
        if self.voice not in self._voices:
            raise SystemExit(f"default voice {self.voice!r} is not among the voices this service "
                             f"offers ({', '.join(self._voices)})")
        if warm:
            # Warm the graph so the first real request is not the slow one. A
            # library caller that only wants the object may not want to pay it.
            self.synthesise("Ready.")

    def _pack_voices(self) -> list[str]:
        try:
            names = list(self.kokoro.get_voices())
        except Exception:  # an older kokoro-onnx without the accessor
            names = []
        return sorted(names) or [self.voice]

    def voices(self) -> list[str]:
        """What this service will speak -- the pack, narrowed by any allowlist."""
        return self._voices

    def language_for(self, voice: str) -> str:
        return cfg.language_for(voice, self.lang)

    def synthesise(self, text: str, voice: str | None = None, speed: float | None = None,
                   trim: bool = True, lang: str | None = None) -> np.ndarray:
        # Measurement passes trim=False: trimming can remove a different amount of silence
        # per variant, which misaligns the waveforms being compared.
        chosen = voice or self.voice
        audio, rate = self.kokoro.create(text, voice=chosen,
                                         lang=lang or self.language_for(chosen),
                                         speed=self.speed if speed is None else speed, trim=trim)
        if rate != RATE or not len(audio) or not np.isfinite(audio).all():
            raise ValueError("synthesis produced invalid audio")
        return np.asarray(audio, np.float32)


def build_handler(engine: Engine, settings):
    max_chars = settings.max_chars
    wait_seconds = settings.wait_seconds
    default_format = settings.default_format
    api_key = settings.api_key
    allow_origin = settings.allow_origin

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "kokoro-pi"

        def log_message(self, fmt, *args):
            # Off by default and never the text: a speech log is a transcript of
            # somebody's day. On, it is method, route and status, nothing else.
            if settings.log_requests:
                super().log_message(fmt, *args)

        def cors(self):
            if allow_origin:
                self.send_header("Access-Control-Allow-Origin", allow_origin)
                self.send_header("Vary", "Origin")

        def authorised(self) -> bool:
            """Constant-time comparison, because a timing oracle on a shared key is real."""
            if not api_key:
                return True
            header = self.headers.get("Authorization", "")
            offered = header[7:] if header.lower().startswith("bearer ") else self.headers.get("X-API-Key", "")
            return hmac.compare_digest(str(offered), str(api_key))

        def refuse(self):
            self.reply(401, {"error": "missing or wrong API key"})
            self.close_connection = True

        def busy(self):
            payload = json.dumps({"error": "busy synthesising; retry shortly"}).encode()
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.cors()
            self.send_header("Retry-After", "1")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def reply(self, status: int, value: dict):
            payload = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.cors()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_OPTIONS(self):
            self.send_response(204)
            self.cors()
            if allow_origin:
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")
                self.send_header("Access-Control-Max-Age", "86400")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            route = urlparse(self.path)
            # /healthz stays open and says nothing useful to a stranger, so a
            # monitor does not need the key and a probe does not leak the setup.
            if route.path == "/healthz":
                self.reply(200, {"ready": True})
                return
            if route.path.startswith("/v1") or route.path == "/readyz":
                if not self.authorised():
                    self.refuse()
                    return
            if route.path == "/readyz":
                self.reply(200, {"ready": True, "backend": engine.backend, "variant": engine.variant,
                                 "voice": engine.voice, "voices": len(engine.voices()),
                                 "lang": engine.lang, "threads": engine.threads,
                                 "format": default_format, "stream": settings.stream,
                                 "wait_seconds": wait_seconds, "busy": engine.lock.locked()})
            elif route.path == "/v1/models":
                self.reply(200, oai.models(engine.variant))
            elif route.path == "/v1/voices":
                self.reply(200, {"voices": engine.voices(), "default": engine.voice,
                                 "languages": {name: engine.language_for(name) for name in engine.voices()}})
            elif route.path == "/v1/tts":
                query = parse_qs(route.query)
                self.speak({key: value[0] for key, value in query.items()})
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self):
            route = urlparse(self.path).path
            if route not in ("/v1/tts", "/v1/audio/speech"):
                self.reply(404, {"error": "not found"})
                return
            if not self.authorised():
                self.refuse()
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1 << 16:
                    raise ValueError("invalid body length")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("body must be an object")
            except (ValueError, TypeError, json.JSONDecodeError):
                self.reply(400, {"error": "expected a JSON object with a 'text' field"})
                self.close_connection = True
                return
            if route == "/v1/audio/speech":
                self.speak_openai(body)
            else:
                self.speak(body)

        def speak_openai(self, request: dict):
            """OpenAI's shape, translated into ours. See openai.py for the two bends."""
            text = request.get("input")
            if not isinstance(text, str) or not text.strip():
                self.reply(400, {"error": {"message": "input is required", "type": "invalid_request_error"}})
                return
            if len(text) > max_chars:
                self.reply(413, {"error": {"message": f"input longer than {max_chars} characters",
                                           "type": "invalid_request_error"}})
                return
            fmt = str(request.get("response_format", "mp3")).lower()
            if fmt not in oai.FORMATS:
                self.reply(400, {"error": {"message": f"response_format must be one of {sorted(oai.FORMATS)}",
                                           "type": "invalid_request_error"}})
                return
            voice = oai.resolve_voice(request.get("voice"), engine.voices(), engine.voice)
            if voice is None:
                self.reply(400, {"error": {"message": f"unknown voice {request.get('voice')!r}; "
                                                      f"see GET /v1/voices",
                                           "type": "invalid_request_error"}})
                return
            speed, clamped = oai.clamp_speed(request.get("speed", engine.speed))

            acquired = (engine.lock.acquire(blocking=False) if wait_seconds <= 0
                        else engine.lock.acquire(timeout=wait_seconds))
            if not acquired:
                self.busy()
                return
            try:
                # Buffered, not streamed: the OpenAI clients that matter read a
                # whole body, and a compressed format cannot be produced
                # incrementally anyway.
                pieces = split_for_streaming(text, settings.stream_limit)
                audio = np.concatenate([engine.synthesise(piece, voice, speed) for piece in pieces])
                if fmt == "pcm":
                    # OpenAI's `pcm` is 24 kHz signed 16-bit little-endian mono,
                    # which is exactly what this model produces.
                    payload, content_type, fell_back = to_bytes(audio, "s16le"), "audio/pcm", False
                else:
                    payload, content_type, fell_back = oai.encode(wav_bytes(audio), fmt)
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.cors()
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Kokoro-Backend", engine.backend)
                self.send_header("X-Kokoro-Voice", voice)
                if fell_back:
                    self.send_header("X-Kokoro-Format-Fallback",
                                     f"{fmt} needs ffmpeg, which is not installed; sent wav")
                if clamped:
                    self.send_header("X-Kokoro-Speed-Clamped", f"{speed}")
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionError, TimeoutError):
                self.close_connection = True
            except Exception as error:
                self.reply(500, {"error": {"message": f"synthesis failed: {type(error).__name__}",
                                           "type": "server_error"}})
            finally:
                engine.lock.release()

        def speak(self, request: dict):
            text = request.get("text")
            if not isinstance(text, str) or not text.strip():
                self.reply(400, {"error": "text is required"})
                return
            if len(text) > max_chars:
                self.reply(413, {"error": f"text longer than {max_chars} characters"})
                return
            layout = str(request.get("format", default_format)).lower()
            if layout not in FORMATS:
                self.reply(400, {"error": f"format must be one of {sorted(FORMATS)}"})
                return
            voice = request.get("voice") or engine.voice
            if voice not in engine.voices():
                self.reply(400, {"error": f"unknown voice {voice!r}; see /v1/voices"})
                return
            try:
                speed = float(request.get("speed", engine.speed))
            except (TypeError, ValueError):
                self.reply(400, {"error": "speed must be a number"})
                return
            if not 0.5 <= speed <= 2.0:
                self.reply(400, {"error": "speed must be between 0.5 and 2.0"})
                return
            lang = request.get("lang") or engine.language_for(voice)
            if not LANG.match(str(lang)):
                self.reply(400, {"error": "lang must be a phonemiser code such as en-us or en-gb"})
                return
            streaming = str(request.get("stream", settings.stream)).lower() not in ("false", "0", "no")

            # 429 rather than 503: this means "occupied, come back", not "broken".
            # wait_seconds of 0 fails fast, which suits callers that queue themselves.
            acquired = (engine.lock.acquire(blocking=False) if wait_seconds <= 0
                        else engine.lock.acquire(timeout=wait_seconds))
            if not acquired:
                self.busy()
                return
            started = time.perf_counter()
            committed = False
            try:
                pieces = (split_for_streaming(text, settings.stream_limit) if streaming
                          else [text.strip()])
                if layout == "wav" or not streaming:
                    # A complete file needs its length up front, so buffer it.
                    audio = np.concatenate([engine.synthesise(piece, voice, speed, lang=lang)
                                            for piece in pieces])
                    payload = wav_bytes(audio) if layout == "wav" else to_bytes(audio, layout)
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav" if layout == "wav"
                                     else f"audio/L16; rate={RATE}; channels=1")
                    self.cors()
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("X-Kokoro-Backend", engine.backend)
                    self.send_header("X-Audio-Seconds", f"{audio.size / RATE:.3f}")
                    self.end_headers()
                    committed = True
                    self.wfile.write(payload)
                    return

                self.send_response(200)
                self.send_header("Content-Type", f"audio/L16; rate={RATE}; channels=1"
                                 if layout == "l16" else "application/octet-stream")
                self.send_header("X-Audio-Format", "s16le" if layout == "s16le" else "s16be")
                self.send_header("X-Audio-Rate", str(RATE))
                self.send_header("X-Kokoro-Backend", engine.backend)
                self.send_header("X-Stream-Pieces", str(len(pieces)))
                self.cors()
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                committed = True
                for piece in pieces:
                    block = to_bytes(engine.synthesise(piece, voice, speed, lang=lang), layout)
                    for start in range(0, len(block), 8192):
                        part = block[start:start + 8192]
                        self.wfile.write(f"{len(part):X}\r\n".encode() + part + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionError, TimeoutError):
                self.close_connection = True
            except Exception as error:  # never leak internals, never hang the client
                if committed:
                    self.close_connection = True
                else:
                    self.reply(500, {"error": f"synthesis failed: {type(error).__name__}"})
            finally:
                engine.lock.release()

    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="kokoro-pi serve", description="Resident Kokoro speech service",
        epilog="Every option can also be set in a config file or the environment: "
               "see docs/configuration.md, or run `kokoro-pi config` to see what is in force.")
    parser.add_argument("--config", help="path to a TOML config file")
    parser.add_argument("--print-config", action="store_true",
                        help="print the effective configuration and exit")
    # Kept because installs in the field pass it; --no-verify is the spelling now.
    parser.add_argument("--skip-verify", dest="verify", action="store_false", default=None,
                        help=argparse.SUPPRESS)
    cfg.add_arguments(parser, cfg.SERVE)
    args = parser.parse_args(argv)

    printing = args.print_config
    del args.print_config
    settings = cfg.resolve(cfg.SERVE, args, "serve", args.config)
    if printing:
        print("kokoro-pi serve\n")
        print(cfg.describe(settings, cfg.SERVE))
        return
    if settings.models is None:
        raise cfg.ConfigError(
            "no models directory. Pass --models, set KOKORO_PI_MODELS, or put\n"
            '  models = "~/.kokoro-pi/models"\n'
            f"  in one of: {', '.join(str(path) for path in cfg.config_paths())}")

    engine = Engine(settings.models, settings.variant, settings.threads, settings.voice,
                    verify=settings.verify, allowed=settings.voices, lang=settings.lang,
                    speed=settings.speed)
    handler = build_handler(engine, settings)
    if settings.wyoming:
        from . import wyoming

        wyoming.serve(engine, settings, split_for_streaming, settings.host, settings.wyoming_port)
        print(f"wyoming listening on {settings.host}:{settings.wyoming_port} "
              f"-- add it in Home Assistant with Settings > Devices > Add integration > Wyoming",
              flush=True)
    print(f"kokoro-pi ready: variant={engine.variant} backend={engine.backend} "
          f"voice={engine.voice} voices={len(engine.voices())} lang={settings.lang} "
          f"threads={engine.threads} format={settings.default_format} "
          f"http://{settings.host}:{settings.port}", flush=True)
    if settings.host not in ("127.0.0.1", "localhost", "::1") and not settings.api_key:
        print("note: serving beyond loopback with no API key. Set api_key, or put it behind "
              "something that authenticates.", flush=True)
    if settings.allow_origin == "*":
        print("note: Access-Control-Allow-Origin is *, so any web page may call this service.",
              flush=True)
    ThreadingHTTPServer((settings.host, settings.port), handler).serve_forever()


if __name__ == "__main__":
    main()
