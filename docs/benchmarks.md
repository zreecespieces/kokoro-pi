# Benchmarks

All numbers from one Raspberry Pi 5 (4× Cortex-A76 at 2.4 GHz, 8 GB, Debian 13,
64-bit), four threads, voice `af_heart`, ONNX Runtime 1.30.0. Medians, warm model,
measured on held-out text the int8 calibration never saw.

Two quantities matter and they are not the same:

- **Synthesis time** — how long the whole passage takes. Throughput.
- **First audio** — how long before a listener hears anything. Interactivity. Without
  streaming these are identical, because nothing leaves until everything is done.

Four configurations, each measured through the same HTTP endpoint with
`tools/bench_http.py`, so the numbers include everything a client
actually waits for — phonemisation, synthesis and transfer.

**One paragraph, ~12 s of speech**

| Setup | First audio | Full passage | Realtime factor |
|---|---:|---:|---:|
| Kokoro on PyTorch (what a plain `pip install` gives you) | 10.45 s | 20.13 s | 1.54 |
| Kokoro + ONNX Runtime (upstream export) | 12.02 s | 12.02 s | 0.98 |
| + graph rewrites and the Snake kernel | 10.41 s | 10.41 s | 0.85 |
| + the fused int8 kernel | 5.47 s | 5.47 s | 0.45 |
| **+ clause streaming — the default** | **1.59 s** | 5.90 s | 0.51 |

**One sentence, ~4 s of speech**

| Setup | First audio | Full passage | Realtime factor |
|---|---:|---:|---:|
| Kokoro on PyTorch | 6.66 s | 6.66 s | 1.46 |
| Kokoro + ONNX Runtime | 4.15 s | 4.15 s | 1.04 |
| + graph rewrites and the Snake kernel | 3.78 s | 3.78 s | 0.95 |
| + the fused int8 kernel | 2.13 s | 2.13 s | 0.53 |
| **+ clause streaming — the default** | **0.93 s** | 2.27 s | 0.58 |

**A short phrase, ~1.1 s of speech**

| Setup | First audio | Full passage | Realtime factor |
|---|---:|---:|---:|
| Kokoro on PyTorch | 2.68 s | 2.68 s | 1.62 |
| Kokoro + ONNX Runtime | 1.41 s | 1.41 s | 1.27 |
| + graph rewrites and the Snake kernel | 1.27 s | 1.27 s | 1.15 |
| + the fused int8 kernel | 0.66 s | 0.66 s | 0.60 |
| + clause streaming | 0.72 s | 0.72 s | 0.65 |

Reading these honestly:

- **Throughput**: the kernels give **2.0–2.2× over the ONNX export** and **about 3×
  over a PyTorch install**. Realtime factor goes from 1.5 (slower than speech) to 0.45
  (twice as fast as speech), which is the difference between unusable and comfortable.
- **Latency**: streaming is what you feel. For a paragraph it cuts the wait from 5.47 s
  to 1.59 s, and against a raw install, from 10.45 s to 1.59 s — **6.6× sooner**.
- **Streaming costs throughput.** Synthesising clause by clause adds roughly 8% to the
  total (5.47 s → 5.90 s) and slightly changes phrasing, because the model sees clause
  boundaries instead of choosing its own pauses. For a short phrase there is no first
  clause to win, so it is pure overhead — send `"stream": false` when you are generating
  files rather than talking to someone.
- **PyTorch's audio is a little longer** for the same text (13.05 s against 12.27 s), so
  compare its realtime factor rather than its seconds.

The graph rewrites look modest here (1.10–1.15×) because these totals include the text
front end, which they do not touch. On model inference alone they are worth 1.35×.

## Per-layer kernel measurements

The vocoder's convolutions are 64% of run time. One layer, `resblocks.5/convs1.0`
(128 channels, 24001 samples, kernel 11) at 8.65 GFLOP:

| Implementation | Time | Throughput | vs float | Quality cost (all 36 convs) |
|---|---:|---:|---:|---:|
| ONNX Runtime float32 | 83.8 ms | 102 GFLOP/s | 1.0× | — |
| ONNX Runtime `QLinearConv` | 23.9 ms | 336 GOP/s | 3.5× | 4.82 dB ✗ |
| ONNX Runtime `ConvInteger` + fast quantiser | 56.3 ms | — | 1.5× | 1.55 dB |
| **kokoro-pi `QuantConv2d`** | **19.2 ms** | **451 GOP/s** | **4.4×** | **1.55 dB** |

The float kernel is already at ~68% of the Pi's ~154 GFLOP/s FP32 peak, which is why
there is no float kernel in this project — there was nothing to win. Dilation makes no
difference to any of these (1, 3 and 5 measure within 5%), and the 256-channel shape
behaves the same: 54.5 ms float against 13.0 ms int8.

## Quality against speed

Calibrated on `corpus/english.json` (9 utterances), measured on `corpus/heldout.json`
(6 disjoint utterances), against the upstream export:

| Build | Convolutions in int8 | Median log-spectral | Worst | Speedup |
|---|---:|---:|---:|---:|
| Phase floor — any change at all | — | 0.53–1.16 dB | 1.29 dB | — |
| Float only (`--skip-int8`) | 0 | 0.00 dB (115+ dB SNR) | — | 1.35× |
| `--families` minus `resblocks.4` and `.1` | 24 | 1.07 dB | 1.17 dB | 1.43× |
| `--families` minus `resblocks.4` | 30 | 1.20 dB | 1.32 dB | 1.57× |
| **Default, everything eligible** | 36 | 2.17 dB | 2.38 dB | **1.83×** |

`resblocks.4` carries most of the cost by itself. In a blind listen over all six
held-out utterances none of these was distinguishable from the float engine, which is
why the default is the fastest — see [quality.md](quality.md) for why the metric is
reported anyway and why waveform SNR is not used.

## Where the time goes

Profile of the deployed graph, medium utterance, 2824 ms of node time:

| Operator | ms | Share |
|---|---:|---:|
| `Conv` (float) | 1668.5 | 59.1% |
| `QLinearConv` (already int8 upstream) | 248.4 | 8.8% |
| `ConvTranspose` | 162.7 | 5.8% |
| `SnakeFused` (native) | 124.0 | 4.4% |
| `MatMul` | 123.1 | 4.4% |
| `Add` | 101.9 | 3.6% |
| `STFT` | 91.7 | 3.2% |
| `InstanceNormalization` | 89.9 | 3.2% |

By stage: vocoder 89.7%, text encoder and projection 4.8%, acoustic decoder 2.5%,
everything else 3.0%. Generator convolutions by group: `resblocks.5` 518.6 ms,
`resblocks.4` 323.4, `resblocks.2` 323.0, `resblocks.1` 219.6, `resblocks.3` 161.1,
`resblocks.0` 111.4, `ups.1` 92.9, `ups.0` 48.0 — 1815 ms, 64.3% of the run.

## Threads

One synthesis does not use four cores well:

| Threads | Median, one paragraph | Speedup | Efficiency |
|---|---:|---:|---:|
| 1 | 11.71 s | 1.00× | 100% |
| 2 | 6.79 s | 1.72× | 86% |
| 4 | 5.26 s | 2.23× | 56% |

Use 4 anyway — 5.26 s beats 6.79 s and nothing else is asking for the cores. But
the inefficiency is not recoverable by synthesising two clauses at once: that
measures **slower** (0.73–0.86×), whichever way it is arranged, which is
evidence the constraint is memory bandwidth rather than idle cores. The
[roadmap](roadmap.md) has the numbers and what follows from them.

## What is left on the table

The int8 kernel runs at 451 GOP/s against a ~614 GOP/s ceiling, so it is 73%
done and there is no large win left inside it. The remaining gains are in the
model and in memory traffic: a frame-rate vocoder is worth 3–5×, fusing residual
blocks to stop streaming 12 MB tensors is worth 1.2–1.5×, and there is a list of
measured dead ends so nobody repeats them — **[roadmap](roadmap.md)**.

## Reproducing

```bash
# every variant, quality and speed, on your hardware
kokoro-pi validate --models ~/.kokoro-pi/models --variants upstream float int8

# any HTTP endpoint, including whatever you run today
python3 tools/bench_http.py --url http://127.0.0.1:8080/v1/tts
python3 tools/bench_http.py --url http://127.0.0.1:5000/v1/tts --label "my old setup"
```

Numbers vary with thermals and with what else the Pi is doing; these were taken on an
otherwise idle machine with active cooling.
