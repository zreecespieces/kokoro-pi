# Troubleshooting

## The service will not start

```bash
systemctl --user status kokoro-pi
journalctl --user -u kokoro-pi -n 50 --no-pager
```

**`checksum mismatch for <name>`** — an asset changed after it was built. Rebuild:
`kokoro-pi build --models ~/.kokoro-pi/models`. If only the model files are suspect,
delete them and the build will re-download and re-derive.

**`no models.json in <dir>`** — the build never finished. Run it again; it skips the
download if the upstream files are already there and verified, and re-running
`install.sh` skips the whole build when a manifest already exists (set
`KOKORO_PI_REBUILD=1` to force one).

**`ModuleNotFoundError: No module named 'kokoro_pi'`** — the unit's `PYTHONPATH` points
somewhere the service cannot see. The installer copies the source into
`~/.kokoro-pi/src` for exactly this reason, because the unit sets `PrivateTmp=true`
(so a clone under `/tmp` is invisible to it) and `/tmp` is cleared on reboot. Re-run
`install.sh`, or edit the `PYTHONPATH` line in
`~/.config/systemd/user/kokoro-pi.service` to a directory that persists.

**`Domain already set in registry`** — the same custom-op domain was registered twice.
That happens if you pass the operator library to ONNX Runtime yourself *and* let this
project do it. Register it once.

**The service stops when you log out** — linger is not enabled:
`loginctl enable-linger $USER`.

## Requests come back 429

The engine synthesises one request at a time. A 429 carries `Retry-After` and means
"occupied, try again" — retry it rather than treating it as failure. If you would rather
requests queue server-side, raise `--wait-seconds`; if you would rather they fail
immediately so you can queue them yourself, set `--wait-seconds 0`.

## The audio is loud noise

Almost always byte order. This service sends little-endian (`s16le`) unless asked
otherwise, and some clients expect big-endian. Either send `"format": "l16"` per
request, or start the service with `--default-format l16`.

## It is not faster than plain Kokoro

Check which variant is live:

```bash
curl -s http://127.0.0.1:8080/readyz
```

`"variant": "int8"` is the fast one. If you see `float` or `upstream`, the int8 build
either was skipped (`--skip-int8`) or your CPU lacks the dot-product extension.

Check the extension:

```bash
grep -o asimddp /proc/cpuinfo | head -1
```

No output means the int8 kernel falls back to portable scalar code and will be *slower*
than float. Raspberry Pi 5 (Cortex-A76) has it; Pi 4 (Cortex-A72) does **not** —
on a Pi 4, build with `--skip-int8` and use the float model, which still gives the 1.35×
from graph rewrites and the Snake kernel.

Also confirm thread count matches your cores (`--threads 4` on a Pi 5) and that nothing
else is saturating the CPU.

## Audio sounds wrong

First, switch to the float variant, which is inaudibly identical to upstream:

```bash
kokoro-pi serve --models ~/.kokoro-pi/models --variant float
```

If that fixes it, the int8 build is at fault — rebuild leaving the sensitive groups in
float:

```bash
kokoro-pi build --models ~/.kokoro-pi/models \
  --families resblocks.0 resblocks.2 resblocks.3 resblocks.5
```

If the float variant sounds wrong too, the problem is upstream of this project: try
`--variant upstream`, which is the published export with none of our changes. If *that*
is also wrong, the issue is in the text front end (phonemisation) rather than the
vocoder, and `espeakng-loader`/`phonemizer-fork` are the place to look.

## Playback problems

Raw PCM needs the right flags. The stream is 24 kHz, mono, signed 16-bit little-endian:

```bash
aplay -r 24000 -f S16_LE -c 1 -          # ALSA
ffplay -f s16le -ar 24000 -ac 1 -        # ffmpeg
```

If audio plays but stutters, your client is probably buffering: pass `curl -N`, or use
`format=wav` and play the file afterwards.

Crackling at the very start of playback is usually the sound card waking up, not
synthesis. Play a short silence first, or keep the device open.

## `pip install` problems

**`kokoro-pi: command not found` after installing** — the entry point went into the
environment's `bin/`, which is not on your `PATH` unless the environment is active.
`python -m kokoro_pi` always works, and `pipx install kokoro-pi` puts the command on
your `PATH` on purpose.

**`no models.json in ~/.kokoro-pi/models`** — `pip install` brings the code, not the
models. They are derived rather than redistributed, so run `kokoro-pi build` once
(a 177 MB download, then a few minutes of calibration).

**You upgraded the package and the kernel did not change** — expected, and worth
knowing. `kokoro-pi build` copies the operator library *into* the models directory and
records its checksum in the manifest, so a models directory is self-contained and a
package upgrade does not reach into it. To pick up a new kernel, rebuild:

```bash
kokoro-pi build --models ~/.kokoro-pi/models     # or KOKORO_PI_REBUILD=1 ./install.sh
```

The upside of the same design is that upgrading the package can never leave a service
running a library its manifest does not describe.

**It compiled the operators even though you installed a wheel** — that happens when pip
fell back to the source distribution, which it does when no wheel matches your platform.
Check what you got:

```bash
python -c "from kokoro_pi import paths; print(paths.prebuilt_library())"
```

`None` means a source install, and `kokoro-pi build` will compile the kernel itself —
which needs `g++` and is fine, just slower to set up. Wheels are published for
`manylinux_2_28` on aarch64 and x86_64.

## Builds fail to compile

**`g++: command not found`** — `sudo apt install build-essential`.

**Errors about `onnxruntime_cxx_api.h`** — the vendored headers under `native/include`
are for ONNX Runtime 1.23's API (version 23) and the pinned runtime is 1.30. If you
changed the `onnxruntime` pin to something older than 1.23, the library will compile but
fail to load with an API version error. Put the pin back.

**x86_64** — the int8 kernel has no AVX path and falls back to scalar code. The build
still works and the float model is still 1.35× faster than upstream, but there is no
int8 benefit. This project is aimed at ARM64.

## Memory and disk

The build needs about 2.5 GB free: a 177 MB download, two derived models of about the
same size, and a calibration copy that is deleted afterwards. What remains is about
855 MB — 545 MB of models and a 305 MB virtual environment. Steady-state runtime is
roughly 900 MB of RSS with the int8 model resident.

You can reclaim 177 MB by deleting `kokoro-v1.0.fp16.onnx` after the build, but then
`--variant upstream` and any rebuild will re-download it.

If the build is killed part-way, it is safe to run again — completed downloads are
verified and reused.
