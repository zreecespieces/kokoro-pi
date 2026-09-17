"""Fetch and verify the upstream Kokoro model and voices.

Nothing here is redistributed: the files come from the kokoro-onnx release and are
checked against known digests, so a corrupted or substituted download fails loudly
rather than producing quietly wrong audio.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request

RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

UPSTREAM = {
    "kokoro-v1.0.fp16.onnx": "c1610a859f3bdea01107e73e50100685af38fff88f5cd8e5c56df109ec880204",
    "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
}

# The upstream file is named "fp16" but the graph holds FLOAT/INT8/UINT8 initialisers and
# QLinear operators - there are no float16 initialisers in it. The name is kept because
# that is what the release calls it; the precision description is in the README.


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify(path: Path, expected: str) -> None:
    actual = digest(path)
    if actual != expected:
        raise SystemExit(f"checksum mismatch for {path.name}\n  expected {expected}\n  found    {actual}")


def download(name: str, destination: Path, expected: str | None = None) -> Path:
    expected = expected or UPSTREAM[name]
    target = destination / name
    if target.exists():
        if digest(target) == expected:
            print(f"  {name}: already present and verified")
            return target
        print(f"  {name}: present but checksum differs, downloading again")
    destination.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    url = f"{RELEASE}/{name}"
    print(f"  {name}: downloading from {url}")
    with urllib.request.urlopen(url) as response, partial.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0"))
        written = 0
        while chunk := response.read(1 << 20):
            output.write(chunk)
            written += len(chunk)
            if total:
                print(f"\r  {name}: {written / 1e6:7.1f} MB / {total / 1e6:.1f} MB", end="", flush=True)
        print()
    verify(partial, expected)
    partial.replace(target)
    return target


def fetch_all(destination: Path) -> dict[str, Path]:
    print(f"upstream assets -> {destination}")
    return {name: download(name, destination, expected) for name, expected in UPSTREAM.items()}
