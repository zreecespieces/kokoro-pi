#!/usr/bin/env bash
# One-command install for kokoro-pi.
#
#   curl -fsSL https://raw.githubusercontent.com/zreecespieces/kokoro-pi/main/install.sh | bash
#
# or, from a clone:  ./install.sh
#
# Creates a virtual environment, downloads the upstream Kokoro model, compiles the
# native operators for this CPU, derives the optimised models, and installs a user
# systemd service. Everything lands under ~/.kokoro-pi unless told otherwise.
set -euo pipefail

REPO=${KOKORO_PI_REPO:-https://github.com/zreecespieces/kokoro-pi.git}
PREFIX=${KOKORO_PI_PREFIX:-$HOME/.kokoro-pi}
PORT=${KOKORO_PI_PORT:-8080}
HOST=${KOKORO_PI_HOST:-127.0.0.1}
VOICE=${KOKORO_PI_VOICE:-af_heart}
THREADS=${KOKORO_PI_THREADS:-$(nproc 2>/dev/null || echo 4)}
SKIP_SERVICE=${KOKORO_PI_SKIP_SERVICE:-0}
SERVE_ARGS=${KOKORO_PI_SERVE_ARGS:-}   # extra flags for the service, e.g. "--default-format l16"
BUILD_ARGS=${KOKORO_PI_BUILD_ARGS:-}
# Everything the service reads lives here afterwards, editable without touching
# the unit. Existing files are never overwritten.
CONFIG=${KOKORO_PI_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/kokoro-pi/config.toml}

say() { printf '\n\033[1;36m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$1" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$1" >&2; exit 1; }

say "checking this machine"
case "$(uname -s)" in
  Linux) ;;
  Darwin) warn "macOS works for building and testing, but the systemd service step is skipped."; SKIP_SERVICE=1 ;;
  *) die "unsupported platform $(uname -s)" ;;
esac
case "$(uname -m)" in
  aarch64|arm64) ;;
  *) warn "not ARM64: the int8 kernel has no vector path here and will be slow. The float model still works." ;;
esac
if [ "$(uname -s)" = "Linux" ] && ! grep -qi asimddp /proc/cpuinfo; then
  warn "this CPU does not report the dot-product extension (asimddp); expect no int8 speedup."
fi
command -v git >/dev/null || die "git is required"
command -v g++ >/dev/null || die "g++ is required (sudo apt install build-essential)"
PYTHON=${PYTHON:-python3}
command -v "$PYTHON" >/dev/null || die "python3 is required"
"$PYTHON" - <<'PY' || die "python 3.11 or newer is required"
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
AVAILABLE=$(df -Pk "$HOME" | awk 'NR==2 {print int($4/1024)}')
[ "$AVAILABLE" -lt 2500 ] && warn "only ${AVAILABLE} MB free in $HOME; the build needs about 2.5 GB."
echo "  $(uname -m) $(uname -s), ${THREADS} threads, ${AVAILABLE} MB free"

SOURCE=$PREFIX/src
if [ -f "$(dirname "$0")/src/kokoro_pi/build.py" ]; then
  CLONE=$(cd "$(dirname "$0")" && pwd)
  case "$CLONE/" in
    "$PREFIX"/*)
      SOURCE=$CLONE
      say "installing from this clone: $SOURCE"
      ;;
    *)
      # The service must not depend on a directory that can disappear or that systemd
      # hides from it: PrivateTmp gives the unit its own /tmp, so a clone there is
      # invisible, and /tmp is cleared on reboot anyway.
      say "copying this clone into $SOURCE so the service has a stable path"
      mkdir -p "$SOURCE"
      tar -C "$CLONE" --exclude .git --exclude models --exclude venv -cf - . | tar -C "$SOURCE" -xf -
      ;;
  esac
else
  say "fetching kokoro-pi"
  mkdir -p "$PREFIX"
  if [ -d "$SOURCE/.git" ]; then
    git -C "$SOURCE" pull --ff-only
  else
    git clone --depth 1 "$REPO" "$SOURCE"
  fi
fi

say "creating the virtual environment"
VENV=$PREFIX/venv
"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip wheel
"$VENV/bin/pip" install --quiet -r "$SOURCE/requirements.txt"
echo "  $("$VENV/bin/python" -c 'import onnxruntime, numpy; print("onnxruntime", onnxruntime.__version__, "numpy", numpy.__version__)')"

MODELS=$PREFIX/models
if [ -f "$MODELS/models.json" ] && [ -z "${KOKORO_PI_REBUILD:-}" ]; then
  say "models already built in $MODELS (set KOKORO_PI_REBUILD=1 to rebuild)"
  echo "  the service verifies every asset's checksum at startup, so a damaged build fails loudly"
else
  say "building models (this is the slow part: a 177 MB download, then a few minutes of CPU)"
  PYTHONPATH="$SOURCE/src" "$VENV/bin/python" -m kokoro_pi build \
    --models "$MODELS" --voice "$VOICE" --threads "$THREADS" $BUILD_ARGS
fi

say "writing the configuration"
if [ -f "$CONFIG" ]; then
  echo "  $CONFIG already exists, leaving it alone"
else
  mkdir -p "$(dirname "$CONFIG")"
  cat > "$CONFIG" <<EOF
# kokoro-pi configuration. Restart the service after editing:
#   systemctl --user restart kokoro-pi
# Every setting, and what it does: https://github.com/zreecespieces/kokoro-pi/blob/main/docs/configuration.md
# See what is actually in force: kokoro-pi config

models = "$MODELS"
host   = "$HOST"
port   = $PORT
voice  = "$VOICE"
threads = $THREADS

# Restrict which voices this service will speak. Empty means all 54.
# voices = ["af_heart", "bf_emma"]

# "auto" takes the language from the voice's name, so bf_emma gets British
# phonemes. Set it explicitly to read any voice any language.
# lang = "auto"

# Bind beyond loopback and this is the difference between a speech service and
# an open one.
# api_key = "change-me"

# Let a browser page call this service directly.
# allow_origin = "*"
EOF
  echo "  wrote $CONFIG"
fi

if [ "$SKIP_SERVICE" = "1" ]; then
  say "done (service step skipped)"
  echo "  start it yourself with:"
  echo "    PYTHONPATH=$SOURCE/src $VENV/bin/python -m kokoro_pi serve --config $CONFIG $SERVE_ARGS"
  exit 0
fi

say "installing the user service"
UNIT_DIR=$HOME/.config/systemd/user
mkdir -p "$UNIT_DIR"
sed -e "s|@VENV@|$VENV|g" -e "s|@SOURCE@|$SOURCE|g" -e "s|@MODELS@|$MODELS|g" \
    -e "s|@CONFIG@|$CONFIG|g" -e "s|@SERVE_ARGS@|$SERVE_ARGS|g" \
    "$SOURCE/systemd/kokoro-pi.service" > "$UNIT_DIR/kokoro-pi.service"
systemctl --user daemon-reload
systemctl --user enable --now kokoro-pi.service
# Without linger the service stops when the login session ends.
loginctl enable-linger "$USER" 2>/dev/null || warn "could not enable linger; the service may stop on logout"

for _ in $(seq 1 60); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

say "ready"
curl -fsS "http://127.0.0.1:$PORT/readyz" && echo
cat <<EOF

  Speak something:
    curl -s -X POST http://127.0.0.1:$PORT/v1/tts \\
      -H 'Content-Type: application/json' -d '{"text":"Hello from my Raspberry Pi."}' \\
      | aplay -q -r 24000 -f S16_LE -c 1 -

  Save a WAV:
    curl -s "http://127.0.0.1:$PORT/v1/tts?text=Hello&format=wav" -o hello.wav

  Measure it here:
    PYTHONPATH=$SOURCE/src $VENV/bin/python -m kokoro_pi validate --models $MODELS

  Configure: $CONFIG
             PYTHONPATH=$SOURCE/src $VENV/bin/python -m kokoro_pi config
  Service:   systemctl --user status kokoro-pi
  Logs:      journalctl --user -u kokoro-pi -f
EOF
