#!/usr/bin/env python3
"""Streaming client that reports when the first audio actually arrives.

    python3 client.py "The first sentence arrives early. The rest follows behind it."
"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/tts"
RATE = 24000


def speak(text: str, url: str = URL) -> bytes:
    body = json.dumps({"text": text, "format": "s16le"}).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    chunks, first = [], None
    with urllib.request.urlopen(request, timeout=300) as response:
        print(f"backend: {response.headers.get('X-Kokoro-Backend')}, "
              f"pieces: {response.headers.get('X-Stream-Pieces')}")
        while chunk := response.read(4096):
            if first is None:
                first = time.perf_counter() - started
            chunks.append(chunk)
    audio = b"".join(chunks)
    total = time.perf_counter() - started
    seconds = len(audio) / 2 / RATE
    print(f"first audio after {first:.2f}s, all {seconds:.2f}s of speech in {total:.2f}s "
          f"(realtime factor {total / seconds:.2f})")
    return audio


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "The first sentence arrives early. The rest follows behind it."
    data = speak(text)
    # Pipe it somewhere useful, e.g. `python3 client.py "..." > out.raw`
    if not sys.stdout.isatty():
        sys.stdout.buffer.write(data)
