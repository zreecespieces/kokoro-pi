#!/usr/bin/env python3
"""Benchmark any speech HTTP endpoint the way a listener experiences it.

Reports time to first audio and total time, so a streaming service and a
synthesise-then-send service can be compared on the same axis. Used to produce the
comparison table in docs/benchmarks.md, and useful for measuring whatever you are
running today before you switch.

    python3 tools/bench_http.py --url http://127.0.0.1:8080/v1/tts
    python3 tools/bench_http.py --url http://127.0.0.1:8000/v1/tts --label "my old setup"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import urllib.request

RATE = 24000
TEXTS = {
    "short": "Switched to Flint.",
    "medium": "I am checking the board now. There are several issues that need your attention.",
    "long": ("The image duplication issue needs attention. The next issue concerns the layout "
             "of the class view. We can investigate the first problem, verify the fix, and then "
             "move on to the remaining work."),
}


def once(url: str, text: str, timeout: float, payload_key: str,
         audio_format: str | None = None, stream: bool = True) -> dict:
    # Send only what was asked for: some endpoints reject bodies with unknown fields.
    request_body: dict = {payload_key: text}
    if audio_format:
        request_body["format"] = audio_format
    if not stream:
        request_body["stream"] = False
    body = json.dumps(request_body).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first = None
    size = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        backend = response.headers.get("X-Kokoro-Backend")
        while chunk := response.read(4096):
            if first is None:
                first = time.perf_counter() - started
            size += len(chunk)
    total = time.perf_counter() - started
    seconds = size / 2 / RATE
    return {"first_audio_seconds": first or total, "total_seconds": total,
            "audio_seconds": seconds, "realtime_factor": total / seconds if seconds else float("nan"),
            "backend": backend}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/tts")
    parser.add_argument("--label", default=None, help="name for this endpoint in the output")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--text-key", default="text", help="JSON field the endpoint expects")
    parser.add_argument("--no-stream", action="store_true",
                        help="ask for the whole passage at once, to isolate what streaming buys")
    parser.add_argument("--format", dest="audio_format",
                        help="send a format field; omit it for endpoints that reject extra fields")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    label = args.label or args.url
    print(f"=== {label}")
    # Every response is 16-bit mono at 24 kHz, so byte counts convert to seconds either way.
    rows = {}
    for name, text in TEXTS.items():
        samples = [once(args.url, text, args.timeout, args.text_key, args.audio_format,
                        not args.no_stream) for _ in range(args.repeats)]
        row = {
            "chars": len(text),
            "audio_seconds": samples[0]["audio_seconds"],
            "backend": samples[0]["backend"],
            "first_audio_seconds": statistics.median(s["first_audio_seconds"] for s in samples),
            "total_seconds": statistics.median(s["total_seconds"] for s in samples),
            "realtime_factor": statistics.median(s["realtime_factor"] for s in samples),
        }
        rows[name] = row
        print(f"  {name:7} {row['audio_seconds']:5.2f}s audio  first {row['first_audio_seconds']:5.2f}s  "
              f"total {row['total_seconds']:5.2f}s  rtf {row['realtime_factor']:.3f}")
    if args.out:
        args.out.write_text(json.dumps({"label": label, "url": args.url, "results": rows}, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
