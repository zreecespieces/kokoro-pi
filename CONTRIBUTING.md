# Contributing

Issues and pull requests are welcome. A few things that will make review quick.

## Claims need measurements

This project exists because of measurements, and its README makes specific numeric
claims. If a change affects speed or quality, include the output of:

```bash
kokoro-pi validate --models ~/.kokoro-pi/models --variants upstream float int8
```

before and after, and say what hardware it ran on.

## Quality is judged spectrally

Do not use waveform SNR to argue that vocoder output is unchanged — for this model it
saturates around 22 dB for any change at all, including pure float rounding. See
[docs/quality.md](docs/quality.md). Waveform SNR is the right gate for the float graph
rewrites only, because those leave the harmonic path bit-identical.

## Things that would be genuinely useful

- **An int16 activation path in the kernel.** 16-bit activations measured 0.68 dB
  against a ~1 dB floor, so a mixed build — int16 for `resblocks.1` and `resblocks.4`,
  int8 elsewhere — should land near 1.8× at floor-level quality.
- **ConvTranspose in int8.** The three upsampling layers are 5% of run time and stay in
  float today because the kernel has no transposed form.
- **`conv_post` support.** It has 22 output channels, and the kernel requires a multiple
  of four. Padding the packed weights would pick up another 0.4%.
- **Other ARM targets.** Anything with `asimddp` should work; Pi 4 does not have it.
  Reports from Orange Pi, Rock 5, Jetson and friends are welcome.
- **More voices and languages in the corpora.** Calibration currently uses nine English
  utterances. Other voices work, but their activation ranges have not been checked.

## Style

Match what is there: readable names, comments that explain *why* rather than restate the
code, and no new dependencies without a reason. The Python targets 3.11+ and the C++
targets C++17.
