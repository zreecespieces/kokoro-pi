#!/usr/bin/env python3
"""Measure this against whatever else you are considering, on your own hardware.

    python3 tools/compare.py --kokoro-pi http://127.0.0.1:8080 \
                             --openai http://127.0.0.1:8880 --label openai=Kokoro-FastAPI \
                             --piper "/tmp/piper-venv/bin/python -m piper -m /tmp/voices/en_US-lessac-medium.onnx"

Every engine is measured the same way: the same text, the same number of
repeats, wall clock from "ask" to "first byte of audio" and to "last byte",
divided by the duration of the audio that came back. Nothing is taken on trust
from a README, including ours.

Realtime factor is the honest headline. Seconds are not comparable between
engines because they do not all produce the same amount of audio for the same
words -- a model that speaks faster finishes sooner while doing the same work.
"""
from __future__ import annotations

import argparse
import json
import shlex
import statistics
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

TEXTS = {
    "phrase": "Your coffee is ready.",
    "sentence": "The vocoder is where almost all of the time goes, so that is the only "
                "part worth optimising.",
    "paragraph": "The vocoder is where almost all of the time goes, so that is the only part "
                 "worth optimising. Everything else is noise, and measuring it carefully is "
                 "what keeps a clever idea from quietly costing more than it saves. The "
                 "convolutions run at four hundred and fifty one billion operations a second, "
                 "which is most of what this processor can do at all.",
}


def audio_seconds(data: bytes, assume_rate: int = 24000) -> float:
    """Duration of a WAV, or of raw 16-bit mono at `assume_rate`."""
    if data[:4] == b"RIFF" and len(data) > 44:
        try:
            with wave.open(__import__("io").BytesIO(data)) as source:
                return source.getnframes() / source.getframerate()
        except wave.Error:
            pass
    if data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return mp3_seconds(data)
    return len(data) / 2 / assume_rate


def mp3_seconds(data: bytes) -> float:
    """Enough MPEG-1 Layer III frame parsing to get a duration without ffprobe."""
    RATES = {0: 44100, 1: 48000, 2: 32000}
    BITRATES = [None, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, None]
    i, total = 0, 0.0
    if data[:3] == b"ID3":
        size = int.from_bytes(data[6:10], "big")
        # Syncsafe: seven bits per byte.
        size = ((data[6] & 0x7F) << 21) | ((data[7] & 0x7F) << 14) | ((data[8] & 0x7F) << 7) | (data[9] & 0x7F)
        i = 10 + size
    while i + 4 <= len(data):
        if data[i] != 0xFF or (data[i + 1] & 0xE0) != 0xE0:
            i += 1
            continue
        rate = RATES.get((data[i + 2] >> 2) & 0x03)
        bitrate = BITRATES[(data[i + 2] >> 4) & 0x0F]
        if not rate or not bitrate:
            i += 1
            continue
        padding = (data[i + 2] >> 1) & 0x01
        length = int(144000 * bitrate / rate) + padding
        total += 1152 / rate
        i += max(length, 1)
    return total


def http_once(url: str, body: dict, headers: dict) -> tuple[float, float, bytes]:
    """Returns seconds to first audio byte, seconds to last, and the audio."""
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **headers})
    started = time.perf_counter()
    first = None
    chunks = []
    with urllib.request.urlopen(request, timeout=600) as response:
        while chunk := response.read(4096):
            if first is None:
                first = time.perf_counter() - started
            chunks.append(chunk)
    return first or 0.0, time.perf_counter() - started, b"".join(chunks)


def piper_once(command: str, text: str) -> tuple[float, float, bytes]:
    """Piper is a process, not a service, so this includes its startup cost."""
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "piper.wav"
        started = time.perf_counter()
        subprocess.run([*shlex.split(command), "-f", str(out)], input=text.encode(),
                       capture_output=True, check=True, timeout=600)
        elapsed = time.perf_counter() - started
        data = out.read_bytes()
    # No streaming from the CLI: first audio is when the file is finished.
    return elapsed, elapsed, data


def measure(name: str, run, repeats: int) -> dict:
    firsts, totals, seconds = [], [], 0.0
    for _ in range(repeats):
        first, total, data = run()
        firsts.append(first)
        totals.append(total)
        seconds = audio_seconds(data)
    return {
        "engine": name,
        "first_audio": statistics.median(firsts),
        "total": statistics.median(totals),
        "audio_seconds": seconds,
        "rtf": statistics.median(totals) / seconds if seconds else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare speech engines on this machine")
    parser.add_argument("--kokoro-pi", help="base URL of a kokoro-pi service")
    parser.add_argument("--openai", action="append", default=[],
                        help="base URL of any OpenAI-compatible speech service; repeatable")
    parser.add_argument("--piper", help="command that runs piper, reading text on stdin")
    parser.add_argument("--label", action="append", default=[],
                        help="rename an engine in the output, e.g. --label openai=Kokoro-FastAPI")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--texts", nargs="*", default=list(TEXTS), choices=list(TEXTS))
    parser.add_argument("--out", type=Path, help="write the raw results as JSON")
    args = parser.parse_args()

    labels = dict(pair.split("=", 1) for pair in args.label)
    engines: list[tuple[str, object]] = []
    if args.kokoro_pi:
        base = args.kokoro_pi.rstrip("/")
        engines.append((labels.get("kokoro-pi", "kokoro-pi"),
                        lambda text, base=base: http_once(f"{base}/v1/tts",
                                                          {"text": text, "format": "wav",
                                                           **({"voice": args.voice} if args.voice else {})}, {})))
        engines.append((labels.get("kokoro-pi-stream", "kokoro-pi (streaming)"),
                        lambda text, base=base: http_once(f"{base}/v1/tts",
                                                          {"text": text, "format": "s16le",
                                                           **({"voice": args.voice} if args.voice else {})}, {})))
    for index, url in enumerate(args.openai):
        base = url.rstrip("/")
        name = labels.get("openai" if index == 0 else f"openai{index}", base)
        engines.append((name, lambda text, base=base: http_once(
            f"{base}/v1/audio/speech",
            {"model": "kokoro", "input": text, "response_format": "wav",
             **({"voice": args.voice} if args.voice else {})}, {})))
    if args.piper:
        engines.append((labels.get("piper", "Piper"),
                        lambda text, command=args.piper: piper_once(command, text)))

    if not engines:
        raise SystemExit("nothing to compare: pass --kokoro-pi, --openai or --piper")

    results: dict[str, list[dict]] = {}
    for key in args.texts:
        text = TEXTS[key]
        print(f"\n## {key} -- {len(text)} characters\n")
        print(f"| Engine | First audio | Full passage | Audio | Realtime factor |")
        print(f"|---|---:|---:|---:|---:|")
        rows = []
        for name, run in engines:
            try:
                row = measure(name, lambda run=run, text=text: run(text), args.repeats)
            except (urllib.error.URLError, subprocess.SubprocessError, OSError) as error:
                print(f"| {name} | — | — | — | failed: {type(error).__name__} |")
                continue
            rows.append(row)
            print(f"| {name} | {row['first_audio']:.2f} s | {row['total']:.2f} s | "
                  f"{row['audio_seconds']:.2f} s | **{row['rtf']:.2f}** |")
        results[key] = rows

    print("\nRealtime factor below 1.0 is faster than speech. Seconds are not comparable "
          "between engines:\nthey do not all produce the same amount of audio for the same "
          "words.")
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
