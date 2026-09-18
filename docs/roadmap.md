# Where the next speedup is

This project is at realtime factor 0.45 on a Raspberry Pi 5 — twice as fast as
speech — and the obvious moves are spent. This file is about what is left, what
it is worth, and what has already been measured and found not to work, so nobody
spends a weekend rediscovering it.

Everything here is measured on a Pi 5 (4× Cortex-A76 at 2.4 GHz, 8 GB, Debian 13)
unless it says estimated. See [benchmarks](benchmarks.md) for the baseline and
[how it works](how-it-works.md) for what is already done.

## The short version

**The kernel is finished. The model is not.**

The fused int8 convolution runs at **451 GOP/s**. One `SDOT` per cycle per core
would cap at 4 × 2.4 GHz × 32 ops = 307 GOP/s, so the measured number is itself
proof the kernel is already dual-issuing; the ceiling is ~614 GOP/s and we are at
**73% of it**. There is at most 1.36× there, in the ideal case, for a great deal
of work.

So further gains have to come from **doing less arithmetic, or moving fewer
bytes** — not from doing the same arithmetic faster. Which puts the interesting
work in the model and in memory traffic.

## Tier 1 — the 4× prize: a frame-rate vocoder

**Worth: 3–5× end to end. Needs training. The only change of this size.**

The vocoder is 89.7% of run time, and it is expensive because of *where* it does
its work: Kokoro's generator upsamples into the time domain and runs its last
residual blocks at nearly the full 24 kHz. Convolutions there are 64% of the
whole run.

A frame-rate vocoder does not do that. Vocos-style heads predict STFT magnitude
and phase at frame rate — about 1/300th the sequence length — and reach the
waveform in a single inverse STFT. The arithmetic drops by roughly an order of
magnitude at comparable quality in the literature, and the whole vocoder stops
being the thing you are waiting for. At a conservative 6× on 89.7% of run time,
total time falls to about a quarter: **RTF ~0.11**, and a Pi 5 could serve
several streams at once.

Three things make this more attractive than it looks:

- **It also unlocks true streaming.** A frame-rate head has no utterance-length
  upsampling stack, so it can be run frame-causally — see Tier 3.
- **It also makes accelerators viable again.** The Hailo-8 result below is
  architectural: one 128×128×11 convolution over 24,001 samples fills all eight
  clusters. Frame-rate tensors are 300× shorter and would fit.
- **It is fine-tuning, not pretraining.** The teacher is the existing vocoder,
  which can generate unlimited paired training data from Kokoro's own acoustic
  features.

The risk is voice fidelity across all 54 voices, and matching the AdaIN style
conditioning. Start with one voice, measure with the existing held-out corpus
and log-spectral distance, and only widen if the single-voice result lands under
the 1.16 dB phase floor.

## Tier 2 — memory traffic, which the measurements now point at

**Worth: 1.2–1.5×. No training. This is where to start without a GPU.**

Two experiments, run for this roadmap, agree on something:

| Threads | Median, one paragraph | Speedup | Efficiency |
|---|---:|---:|---:|
| 1 | 11.71 s | 1.00× | 100% |
| 2 | 6.79 s | 1.72× | 86% |
| 4 | 5.26 s | 2.23× | 56% |

Four threads waste nearly half of two of the cores. The obvious recovery is to
synthesise two clauses at once instead of one clause on four threads — and it
**does not work**:

| Arrangement | Median | vs today |
|---|---:|---:|
| One engine, one clause at a time, 4 threads (today) | 4.60 s | 1.00× |
| One engine, two clauses at once, 2 threads | 6.31 s | 0.73× |
| **Two engines, two clauses at once, 2 threads each** | **5.35 s** | **0.86×** |

The first row of that second table is ONNX Runtime's intra-op pool being
per-session: concurrent `Run()` calls share it, so two requests got two threads
of work done between them, not four. Two separate sessions fix that and it is
*still slower than doing one thing at a time*.

That is the interesting part. If the 56% efficiency at four threads were
parallelisation overhead or a serial fraction, overlapping two independent
synthesis streams would have hidden it. It did not — so the binding constraint
is very likely **memory bandwidth**, not arithmetic or scheduling. Concurrent
streams make bandwidth contention worse, which is exactly what the numbers show.

Which suggests, in order:

1. **Confirm the diagnosis.** Measure 4-thread scaling efficiency against
   utterance length. If short utterances (smaller tensors, better cache
   residency) scale better than long ones, bandwidth is confirmed, cheaply and
   without perf counters.
2. **Fuse and tile whole residual blocks.** At 128 channels × 24,001 samples,
   one activation tensor is 12.3 MB — against 512 KB of L2 per core and 2 MB of
   shared L3. Every elementwise operator therefore streams 12 MB in and 12 MB
   out for a handful of flops per element, and `Snake` + `Add` +
   `InstanceNormalization` are 11.2% of run time doing almost no arithmetic.
   Processing a block `conv → snake → conv → add` in time-domain tiles that stay
   in L2 should remove most of that traffic and improve the convolutions'
   locality at the same time.
3. **Keep activations in int8 between layers.** The kernel currently dequantises
   to float32, so it writes 4 bytes per element where 1 would do — a 4× traffic
   cut is available on both the write and the following read. The reason not to
   is that this is what `QLinearConv` does, and it costs 4.82 dB. But
   `QLinearConv` re-quantises with **one scale for a whole tensor**, which is
   the actual source of that damage; int8 storage with a **per-channel** scale
   vector is a different proposition and untested. The existing calibration
   harness can simulate it in an afternoon and the answer is a number, not an
   opinion. Expect somewhere between 1.55 dB and 4.82 dB — worth knowing where.

## Tier 3 — structural changes to the model

**Worth: 1.2–1.5× each. Needs training.**

- **Channel pruning plus distillation.** Convolution cost scales with
  `C_in × C_out`, so pruning 30% of channels through the generator is ~0.49× the
  arithmetic in the layers that matter — roughly **1.4× end to end** if quality
  holds. With the full model as teacher this is fine-tuning, and the
  quality gate already exists.
- **One fewer upsampling stage** (a larger iSTFT hop). The last three residual
  blocks are 1003 ms of the 2824 ms profile, at twice the temporal resolution of
  the first three. Halving that rate saves ~18%, so **1.22×** — the smallest of
  the training-shaped options, and worth doing only as part of Tier 1.
- **Frame-causal normalisation.** Not a speedup: a latency change. The vocoder
  normalises over the whole utterance at 44 AdaIN sites, which is why streaming
  has to be clause-by-clause and why first audio is 1.59 s rather than
  ~100 ms. Replacing those with something frame-causal — a style-conditioned
  affine, or running statistics — by fine-tuning would give constant-latency,
  constant-memory streaming. Estimating the statistics instead **does not work**;
  see below.

## Tier 4 — quality headroom, not speed

**int16 activations for the sensitive groups.** 16-bit activations everywhere
measured **0.68 dB** against the current build's 2.17 dB. A mixed build — int16
for `resblocks.4` and `resblocks.1`, int8 elsewhere — should land near the
1.16 dB phase floor at roughly today's speed. Needs an `SMLAL` path in the
kernel. Worth being clear that this buys *quality at equal speed*, not speed.

Also outstanding and small: `ConvTranspose` in int8 (5.8% of run time, no
transposed form in the kernel yet) and `conv_post` (0.4%, skipped because 22
output channels is not a multiple of four).

## Not research, but worth more than most of it

- **A phrase cache.** Assistant-shaped workloads say the same sentences
  constantly. A content-addressed cache keyed on `(text, voice, speed, lang)`
  makes the second "Good morning" free. Nothing else on this page can beat
  ∞× on a cache hit.
- **Overclocking.** A Pi 5 at 3.0 GHz is a documented 25% clock bump and this
  workload is not thermally gentle but is short-burst. Worth roughly 1.2× for
  the price of a fan, with the usual stability caveats. Not our work, but our
  users' easiest win.

## Measured dead ends

Numbers so nobody repeats these.

| Idea | Why not |
|---|---|
| **Hailo-8 / Pi AI HAT** | The hot convolution ran 1.70× faster on the accelerator, but it is ~2.2% of run time: **0.9% end to end**. 8-bit on that path cost 20 dB SNR at the layer, and one 128×128×11 convolution fills all eight clusters, so a resident multi-layer partition does not fit. Revisit only after Tier 1. |
| **VideoCore VII GPU** | ~50 GFLOP/s against the CPU's ~154, and no integer dot product. Slower than what we have. |
| **SVE / SME** | Not present on Cortex-A76. The Pi 5 has `asimddp` and that is the instruction we use. |
| **Estimating AdaIN statistics for frame streaming** | A strided pass over **half** the frames biases the statistics 1.26%, which costs 3.5 dB against a ~1 dB floor. Needs a retrained vocoder, not a cleverer estimator. |
| **A hand-written float convolution kernel** | ONNX Runtime's float kernels already hit 102 GFLOP/s = 68% of FP32 peak. Nothing to win. |
| **FFT or Winograd convolution** | For kernel 11, the arithmetic reduction is ~2× in float — less than the 4.4× the int8 kernel already delivers, and it does not compose with int8. |
| **int4 weights** | int8 already costs 2.17 dB against a 1.16 dB floor. Halving again spends quality this model does not have. |
| **Utterance-level parallelism** | Measured above: 0.73–0.86×, i.e. slower. The cores are not idle for want of work. |

## Contributing

The tiers are roughly in order of value, and inversely in order of how easy they
are to start. If you have a GPU and an interest in vocoders, Tier 1 is the one
that matters. If you have a Pi and a C++ compiler, start with Tier 2 step 1 —
it is an afternoon, it either confirms or kills the bandwidth diagnosis, and
everything else in that tier depends on the answer.

Quality is gated on log-spectral distance against a held-out corpus, never on
waveform SNR, and [there is a good reason](quality.md). `kokoro-pi validate`
is the gate; please run it before and after.
