# How it works

Kokoro is an 82M-parameter TTS model. On a Raspberry Pi 5 the published ONNX export
synthesises slower than real time, which makes it awkward for anything interactive.
This project makes the same model about 2.5× faster on the same hardware, and starts
speaking sooner than that, without retraining and without an accelerator.

Everything below was measured on a Raspberry Pi 5 (4× Cortex-A76 at 2.4 GHz, 8 GB,
Debian 13) with ONNX Runtime 1.30 and four threads.

## Where the time goes

Profile the deployed graph per node and the answer is not subtle:

| Stage | Share of run time |
|---|---:|
| **Vocoder** (decoder's generator) | **89.7%** |
| ALBERT text encoder + projection | 4.8% |
| Acoustic decoder | 2.5% |
| Prosody, duration, text encoder, other | 3.0% |

Inside the vocoder, convolutions are **64%** of the whole run — 1815 ms of a 2824 ms
medium-length utterance. So the vocoder's convolutions are the only thing worth
optimising, and everything else is noise.

## Step 1: rewrite the graph, keep the arithmetic

Four algebra-preserving rewrites, applied only to the generator:

1. **Normalisation fusion.** The export spells per-channel temporal normalisation as
   nine nodes (`ReduceMean`, `Sub`, `Mul`, `ReduceMean`, `Add`, `Sqrt`, `Div`, `Mul`,
   `Add`). Since Kokoro is batch-one, that whole chain is exactly
   `InstanceNormalization` with the style vectors as its scale and bias. 42 sites.
2. **Height-one convolutions.** 1-D convolutions become 2-D convolutions with a
   height of one, which makes ONNX Runtime's blocked kernels eligible. 37 sites.
3. **Squares.** `Pow(x, 2)` becomes `Mul(x, x)`. 50 sites.
4. **Snake activation fusion.** Kokoro's activation is
   `x + (1/α)·sin(αx)²`, which the graph spells as five elementwise nodes. At 24 kHz
   each of those tensors is megabytes, so the chain is bound by memory traffic, not
   arithmetic. A single native operator computes it in one pass. 48 sites.

Together these give **1.35×** with the waveform preserved to 115 dB SNR — which is to
say, bit-for-bit inaudible. The rewrites are checked against the upstream model's own
audio during `kokoro-pi build`, and the build refuses to write a manifest if that
check fails.

## Step 2: the convolutions were already near the hardware limit

`resblocks.5/convs1.0` does 8.65 GFLOP in 82.5 ms. That is **102 GFLOP/s**, about
**68% of the Pi's ~154 GFLOP/s FP32 peak**. ONNX Runtime's float kernels are good;
there is no hand-written float kernel worth writing here.

That leaves two options: fewer operations (retraining, out of scope) or lower
precision.

## Step 3: int8, and why the obvious ways don't work

int8 is genuinely fast on this CPU — ARM's dot-product instruction (`SDOT`) does 16
multiply-accumulates per instruction. Measured on the hot shape, 128 channels × 24001
samples, kernel 11:

| Approach | Time | vs float | Quality cost |
|---|---:|---:|---:|
| float32 (ONNX Runtime) | 83.8 ms | 1.0× | — |
| `QLinearConv` (standard int8) | 23.9 ms | 3.5× | **4.82 dB** ✗ |
| `ConvInteger` + fast quantiser | 56.3 ms | 1.5× | 1.55 dB |
| **This project's fused kernel** | **19.2 ms** | **4.4×** | **1.55 dB** |

Two problems with the standard operators:

- **`QLinearConv` must write int8.** Re-quantising every layer's output to 256 levels,
  with one scale for the whole tensor, is what makes a naive int8 build sound wrong.
  It is the fastest standard option and the least usable one.
- **`ConvInteger` keeps int32 accumulation** — good quality — but materialises a 12 MB
  int32 intermediate and then needs separate cast, scale and bias passes. The memory
  traffic eats most of the win.

The activations are also awkward: their peak is **25–50× their standard deviation**, so
a single scale for a whole tensor wastes most of the 256 available levels.

## Step 4: the fused kernel

One operator does all three phases in a single pass, with nothing in between
materialised and no output ever rounded to int8:

1. **Quantise per input channel.** Each channel gets its own scale, which is what the
   heavy tails demand. The quantised values are written in a channel-blocked layout so
   that four consecutive positions of four channels form one contiguous 16-byte SDOT
   operand.
2. **Accumulate in int32 with SDOT.** 16 accumulators cover 16 positions × 4 output
   channels; one weight load plus four activation loads feed 16 `SDOT` instructions.
3. **Dequantise per output channel straight to float**, with the bias folded in.

The trick that makes per-channel activation scales possible: an int32 dot product needs
**one** scale across its whole reduction, so per-channel scales cannot survive it.
Instead they are folded into the weights when the model is built (`v = w · s`, then
quantised per output channel). The runtime sees a plain int8 convolution; the
per-channel behaviour is baked into the weights. That folding costs a little weight
resolution — 1.42 dB becomes 1.55 dB — which is a bargain for what it buys.

Result: **451 GOP/s**, faster than ONNX Runtime's own int8 path (336 GOP/s) and 4.4×
the float kernel, with the quality of the int32-accumulating approach.

## Step 5: streaming, which is a different kind of faster

Everything above is throughput. What you actually perceive is **when speech starts**.
The service splits text at sentence and clause boundaries, synthesises each piece in
order, and writes each piece to the socket as soon as it exists. For a paragraph, the
first audio arrives after the first clause rather than after the whole passage — measured
at 3.3× sooner on the upstream graph and more than that once the kernels are in play.

It is worth being clear about the trade: streaming costs a little **throughput** to buy
a lot of **latency**. Synthesising four clauses separately does slightly more total work
than synthesising one passage — measured at roughly 10-20% more wall time — and the
phrasing changes a little, because the model sees clause boundaries where it would
otherwise have chosen its own pauses. For anything interactive that is an easy trade;
if you are batch-generating files, pass `"stream": false` and keep the throughput.

This is also why there is a resident service at all rather than a CLI: the model stays
loaded, so no request pays the load cost.

## What we tried that didn't work

Worth recording, because the negative results are load-bearing:

- **Hailo-8 offload.** The Pi AI HAT was benchmarked properly. The hot convolution ran
  1.70× faster on the accelerator than on the CPU, but it is only ~2.2% of run time, so
  the end-to-end win was **0.9%** — and 8-bit on that path cost 20 dB SNR at the layer.
  One 128×128×11 convolution also fills all eight of the chip's clusters, so a resident
  multi-layer partition does not fit. **This project has no Hailo dependency.**
- **Frame-level streaming inside the vocoder.** Kokoro's vocoder normalises over the
  whole utterance at 44 AdaIN sites, so short windows need global statistics they cannot
  have. Estimating them fails: even a strided pass over **half** the frames biases the
  statistics 1.26%, which costs 3.5 dB of spectral damage against a ~1 dB floor. Getting
  true frame-causal streaming needs a retrained vocoder, not a clever trick.
- **A hand-written float convolution kernel.** Pointless at 68% of peak, as above.

See [quality.md](quality.md) for how quality is measured — the short version is that
waveform SNR is misleading for this model, and the reason is interesting.
