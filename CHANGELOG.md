# Changelog

Dates are when the work landed on `main`. Versions follow
[semantic versioning](https://semver.org): the HTTP API and the config file keys
are the public surface.

## 1.2.0 — 2026-09-18

Installable from PyPI, and usable as a library in place of kokoro-onnx.

### Added

- **`pip install kokoro-pi`.** Binary wheels for `manylinux_2_28` on aarch64 and
  x86_64 carry a **pre-compiled kernel**, so nothing needs a compiler; anywhere
  else pip falls back to the source distribution and `kokoro-pi build` compiles
  the operators there. The wheel is tagged `py3-none`, not per-Python, because
  the library is loaded by ONNX Runtime rather than imported by CPython.
  Published by GitHub Actions through PyPI trusted publishing — no API token
  exists to leak.
- **`from kokoro_pi import Kokoro`** — kokoro-onnx's API over these kernels.
  `create`, `create_stream` and `get_voices` take the same arguments and return
  the same things, delegated to kokoro-onnx itself so nothing about
  phonemisation or voices is reimplemented. The constructor takes a models
  *directory* rather than model files, because these models are derived rather
  than downloaded.
- `kokoro_pi.paths`, which finds `native/` and `corpus/` whether the package was
  cloned or installed, and reports a pre-compiled kernel when a wheel shipped one.

### Changed

- `--corpus` and `--heldout` default to the packaged corpora rather than to
  paths relative to a repository that may not exist.
- `Engine(..., warm=False)` skips the warm-up synthesis, for library callers who
  only want the object.

## 1.1.0 — 2026-09-18

Speaks other people's protocols, and can be configured without editing a
systemd unit.

### Added

- **`POST /v1/audio/speech`**, the OpenAI speech API, so Open WebUI, LibreChat,
  SillyTavern and the OpenAI SDKs work against this service with a base URL
  change and nothing else. OpenAI's voice names map onto Kokoro's, `mp3`/`opus`/
  `aac`/`flac` are produced with ffmpeg when it is installed, and `GET
  /v1/models` answers for clients that ask first.
- **The Wyoming protocol**, which is how Home Assistant talks to a voice
  service — `wyoming = true`, and every voice appears in Home Assistant tagged
  with its language. It advertises `supports_synthesize_streaming` and
  implements the streaming events, so a sentence from a conversation agent
  starts playing at its first clause.
- **Configuration from a file, the environment or a flag**, resolved in that
  reverse order, with `kokoro-pi config` to print what is in force and where
  each value came from. `install.sh` writes
  `~/.config/kokoro-pi/config.toml` and points the unit at it.
- **`voices`**, an allowlist, so `/v1/voices` is an honest menu of what a
  service will speak and a typo is a 400 rather than a surprise accent.
- **`api_key`** (constant-time, on `/v1` and `/readyz`; `/healthz` stays open
  and says only `{"ready": true}`) and **`allow_origin`** with `OPTIONS`
  preflight, so a browser page can call a Pi.
- **`speed`, `stream`, `stream_limit`, `log_requests`** as service-wide
  defaults. Request logging is off by default and never includes the text.
- **[A page you can hear it on](https://zreecespieces.github.io/kokoro-pi/)**,
  and `tools/compare.py` for measuring this against Piper, Kokoro-FastAPI or
  anything else that speaks the OpenAI API, on your own hardware.
- `tools/check_protocols.py`, which exercises every protocol a running service
  speaks and says what broke.

### Changed

- **`lang` defaults to `auto`, which reads the language from the voice's name.**
  Kokoro names voices `<language><gender>_<name>`, and `bf_emma` was being given
  American phonemes — a British voice pronouncing an American's vowels. If you
  were relying on everything being read as `en-us`, set `lang = "en-us"`.
  Only `en-us` and `en-gb` are tested; espeak's Japanese and Chinese phonemes
  are not the ones Kokoro was trained on.
- `/readyz` reports more: voice count, language mode, format, streaming and
  `wait_seconds`.

### Fixed

- **Re-running `install.sh` to upgrade left the old code running.**
  `systemctl --user enable --now` does nothing to a unit that is already active,
  so the installer rewrote the unit, printed "ready", and the previous process
  kept serving.

## 1.0.0 — 2026-09-17

First public release: the speech engine from a personal assistant, packaged.

- Four algebra-preserving graph rewrites and a fused native Snake activation:
  1.35× with the waveform preserved to 115 dB.
- A hand-written fused int8 convolution kernel using ARM's `SDOT`: **451 GOP/s**,
  faster than ONNX Runtime's own int8 path (336 GOP/s) and 4.4× its float
  kernel, with per-input-channel activation scales folded into the weights.
- Clause-level streaming, so first audio arrives after the first clause instead
  of after the whole passage.
- **2.0–2.2× over the upstream ONNX export end to end**, about 3× over a
  PyTorch install, and first audio 6.6× sooner.
- One-command install, a resident systemd service, 54 voices, three variants
  side by side, checksum-verified assets, disjoint calibration and held-out
  corpora, and quality gated on log-spectral distance rather than waveform SNR.
