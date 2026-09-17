#!/usr/bin/env bash
# Save speech to a WAV file. The GET form is handy for quick tests and browsers.
set -euo pipefail
URL=${KOKORO_PI_URL:-http://127.0.0.1:8080/v1/tts}
TEXT=${1:-Hello from my Raspberry Pi.}
OUT=${2:-hello.wav}
curl -sS -G "$URL" --data-urlencode "text=$TEXT" --data-urlencode "format=wav" -o "$OUT"
echo "wrote $OUT"
