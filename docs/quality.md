# Measuring quality honestly

The headline claim of this project is speed, which means the interesting question is
what it costs. Getting that measurement right turned out to be harder than getting the
speedup, so here is the method and the trap.

## Waveform SNR does not work for this model

The intuitive check — synthesise the same text twice and compare the waveforms — is
actively misleading here. Kokoro's vocoder generates its excitation from a harmonic and
noise path that is **chaotic in phase**. Perturb the vocoder's harmonic input by a
relative 1e-7, which is the size of float32 rounding, and:

| Relative perturbation | Waveform SNR | Log-spectral distance |
|---|---:|---:|
| 1e-7 | **28.9 dB** | 0.53 dB |
| 1e-6 | 23.5 dB | 1.08 dB |
| 1e-5 | 22.7 dB | 1.11 dB |
| 1e-4 | 22.6 dB | 1.12 dB |
| 1e-3 | 22.1 dB | 1.16 dB |

Waveform SNR collapses immediately and then **saturates near 22 dB** no matter how much
larger the perturbation gets, while the spectrum hardly moves. The acoustic input path,
by contrast, is well behaved: the same 1e-7 perturbation there leaves 117 dB.

So for anything that touches the harmonic path, roughly **22 dB SNR / 0.997 correlation
/ ~1 dB log-spectral distance is the floor for "any change at all"**, not evidence of
damage. Two mathematically equivalent implementations that differ only in float
rounding will look 22 dB apart.

Reproduce it yourself — this is not a claim you have to take on faith:

```bash
kokoro-pi validate --models ~/.kokoro-pi/models --variants float int8
```

## What is used instead

- **Log-spectral distance**: mean absolute difference of log-magnitude STFT, in dB.
  Insensitive to the phase reshuffling above, sensitive to actual spectral change.
- **Spectral convergence**: relative L2 of the magnitude spectra, which weights the
  high-energy bins that dominate perception.

Both are compared against the phase floor measured on the same utterance class, so
"at the floor" means "as different from the original as harmless rounding is".

Waveform SNR is still the right gate for the **float** rewrites, because those leave
the harmonic path bit-identical — and it duly reports 115+ dB there. `kokoro-pi build`
uses it for exactly that and refuses to proceed below 60 dB.

## What the int8 build actually costs

Calibrated on the 9 utterances of `corpus/english.json`, measured on the disjoint 6 of
`corpus/heldout.json`:

| Variant | Convolutions in int8 | Median log-spectral | Worst |
|---|---:|---:|---:|
| Phase floor (reference) | — | 0.53–1.16 dB | 1.29 dB |
| `--families resblocks.0 resblocks.2 resblocks.3 resblocks.5` | 24 | **1.07 dB** | 1.17 dB |
| `--families` without `resblocks.4` | 30 | 1.20 dB | 1.32 dB |
| default (everything eligible) | 36 | 2.17 dB | 2.37 dB |

The default is measurably above the floor. In a blind listen across all six held-out
utterances and all three variants, none was distinguishable from the float engine, which
is why the default is the fastest one — but the metric is reported honestly, and if you
want to stay at the floor, build with `--families` and lose about a quarter of the gain.

`resblocks.4` is responsible for most of the damage on its own: excluding it takes the
median from 2.17 dB to 1.20 dB.

## Two things that nearly fooled us

**Calibrating and evaluating on the same utterance.** An early simulation predicted
1.55 dB for the default build; the real held-out number is 2.17 dB. The difference was
entirely that the simulation calibrated activation ranges on the very utterance it then
measured. Always evaluate on text the calibration never saw — that is why this repo
ships two disjoint corpora.

**Reference arithmetic that differs from the kernel's.** The kernel multiplies by a
float32 reciprocal because dividing per element is far slower. A numpy reference that
divides disagrees with it on about one activation value in 200,000, which showed up as a
5.9e-4 mismatch and looked like a kernel bug for a while. It wasn't: against its own
definition the kernel is exact to 0.0. If you write a reference, mirror the reciprocal.
