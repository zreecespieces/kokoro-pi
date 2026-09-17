"""Resident HTTP speech service.

Two things make this faster to *hear* than calling Kokoro directly:

* the optimised graph and native kernels synthesise about 2.5x quicker, and
* long text is split into clauses and each clause is streamed as soon as it is ready,
  so the first audio arrives after the first clause instead of after the whole passage.

The model stays resident, so there is no per-request load cost.
"""
from __future__ import annotations

import argparse
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

RATE = 24000
FORMATS = {"s16le", "l16", "wav"}
SENTENCE = re.compile(r"(?<=[.!?…])\s+|\n+")
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
                 voice: str | None = None, verify: bool = True):
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
        self.synthesise("Ready.")  # warm the graph so the first real request is not the slow one

    def voices(self) -> list[str]:
        try:
            names = list(self.kokoro.get_voices())
        except Exception:  # an older kokoro-onnx without the accessor
            names = []
        return sorted(names) or [self.voice]

    def synthesise(self, text: str, voice: str | None = None, speed: float = 1.0,
                   trim: bool = True) -> np.ndarray:
        # Measurement passes trim=False: trimming can remove a different amount of silence
        # per variant, which misaligns the waveforms being compared.
        audio, rate = self.kokoro.create(text, voice=voice or self.voice, lang="en-us",
                                         speed=speed, trim=trim)
        if rate != RATE or not len(audio) or not np.isfinite(audio).all():
            raise ValueError("synthesis produced invalid audio")
        return np.asarray(audio, np.float32)


def build_handler(engine: Engine, max_chars: int, wait_seconds: float):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "kokoro-pi"

        def log_message(self, *_):
            pass  # spoken text never goes to a log

        def reply(self, status: int, value: dict):
            payload = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            route = urlparse(self.path)
            if route.path in ("/healthz", "/readyz"):
                self.reply(200, {"ready": True, "backend": engine.backend, "variant": engine.variant,
                                 "voice": engine.voice, "threads": engine.threads,
                                 "busy": engine.lock.locked()})
            elif route.path == "/v1/voices":
                self.reply(200, {"voices": engine.voices(), "default": engine.voice})
            elif route.path == "/v1/tts":
                query = parse_qs(route.query)
                self.speak({key: value[0] for key, value in query.items()})
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self):
            if urlparse(self.path).path != "/v1/tts":
                self.reply(404, {"error": "not found"})
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
            self.speak(body)

        def speak(self, request: dict):
            text = request.get("text")
            if not isinstance(text, str) or not text.strip():
                self.reply(400, {"error": "text is required"})
                return
            if len(text) > max_chars:
                self.reply(413, {"error": f"text longer than {max_chars} characters"})
                return
            layout = str(request.get("format", "s16le")).lower()
            if layout not in FORMATS:
                self.reply(400, {"error": f"format must be one of {sorted(FORMATS)}"})
                return
            voice = request.get("voice") or engine.voice
            if voice not in engine.voices():
                self.reply(400, {"error": f"unknown voice {voice!r}"})
                return
            try:
                speed = float(request.get("speed", 1.0))
            except (TypeError, ValueError):
                self.reply(400, {"error": "speed must be a number"})
                return
            if not 0.5 <= speed <= 2.0:
                self.reply(400, {"error": "speed must be between 0.5 and 2.0"})
                return
            streaming = str(request.get("stream", "true")).lower() not in ("false", "0", "no")

            if not engine.lock.acquire(timeout=wait_seconds):
                self.reply(503, {"error": "busy synthesising; retry shortly"})
                return
            started = time.perf_counter()
            committed = False
            try:
                pieces = split_for_streaming(text) if streaming else [text.strip()]
                if layout == "wav" or not streaming:
                    # A complete file needs its length up front, so buffer it.
                    audio = np.concatenate([engine.synthesise(piece, voice, speed) for piece in pieces])
                    payload = wav_bytes(audio) if layout == "wav" else to_bytes(audio, layout)
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav" if layout == "wav"
                                     else f"audio/L16; rate={RATE}; channels=1")
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
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                committed = True
                for piece in pieces:
                    block = to_bytes(engine.synthesise(piece, voice, speed), layout)
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
    parser = argparse.ArgumentParser(prog="kokoro-pi serve", description="Resident Kokoro speech service")
    parser.add_argument("--models", type=Path, required=True, help="directory holding models.json")
    parser.add_argument("--variant", help="int8 (default), float, or upstream")
    parser.add_argument("--voice", help="override the manifest's default voice")
    parser.add_argument("--host", default="127.0.0.1", help="default is loopback only")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--wait-seconds", type=float, default=30.0,
                        help="how long a request waits for the engine before 503")
    parser.add_argument("--skip-verify", action="store_true", help="skip asset checksum verification")
    args = parser.parse_args(argv)

    engine = Engine(args.models, args.variant, args.threads, args.voice, verify=not args.skip_verify)
    handler = build_handler(engine, args.max_chars, args.wait_seconds)
    print(f"kokoro-pi ready: variant={engine.variant} backend={engine.backend} "
          f"voice={engine.voice} threads={engine.threads} http://{args.host}:{args.port}", flush=True)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("note: serving beyond loopback. There is no authentication; put it behind "
              "something that has some.", flush=True)
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
