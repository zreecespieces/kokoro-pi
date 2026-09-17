#!/usr/bin/env bash
# Speak text through a running kokoro-pi service, straight to the speaker.
#   ./say.sh "Hello from my Raspberry Pi."
set -euo pipefail
URL=${KOKORO_PI_URL:-http://127.0.0.1:8080/v1/tts}
TEXT=${*:-Hello from my Raspberry Pi.}

# Audio is streamed clause by clause, so playback starts before synthesis finishes.
curl -sS -N -X POST "$URL" \
  -H 'Content-Type: application/json' \
  -d "$(printf '{"text":%s}' "$(printf '%s' "$TEXT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")" \
  | aplay -q -r 24000 -f S16_LE -c 1 -
