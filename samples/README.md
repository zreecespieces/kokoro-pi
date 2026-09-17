# Audio samples

Same text, same voice, three builds. They should be hard to tell apart — that is the
point of the exercise, since the whole project is about speed at unchanged quality.

| Suffix | What it is |
|---|---|
| `-upstream.wav` | The published Kokoro ONNX export, unmodified |
| `-float.wav` | Graph rewrites + the native Snake kernel (1.35×, inaudibly identical: 115 dB) |
| `-int8.wav` | Adds the fused int8 convolution kernel (the default build) |

Regenerate all of them on your own hardware with:

```bash
make samples     # or: kokoro-pi validate --models ... --audio-dir samples --repeats 1
```

The measured difference between `-upstream` and `-int8` is about 2.2 dB log-spectral
distance, against a floor of roughly 1 dB for any change whatsoever. See
[../docs/quality.md](../docs/quality.md) for what that means and why waveform
comparison is the wrong tool here.
