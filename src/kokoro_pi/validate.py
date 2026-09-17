"""Quality and speed measurement.

Quality is reported **spectrally**, not as waveform SNR. This vocoder's noise
excitation is chaotic in phase: perturbing its harmonic input by one part in ten
million - the size of float32 rounding - already drops waveform SNR to about 28 dB
and saturates near 22 dB for any larger change, while the spectrum barely moves.
Waveform SNR therefore measures phase reshuffling, not audible damage, so the gate
here is log-spectral distance with that floor (roughly 0.5-1.3 dB) as the reference
point. See docs/quality.md.

Waveform SNR *is* meaningful for the float rewrites, which leave the harmonic path
bit-identical, and `build` uses it there.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import statistics
import time
import wave

import numpy as np

RATE = 24000


def stft_magnitude(audio: np.ndarray, size: int = 1024, hop: int = 256) -> np.ndarray:
    audio = np.asarray(audio, np.float64).ravel()
    frames = 1 + max(0, (len(audio) - size) // hop)
    window = np.hanning(size)
    view = np.lib.stride_tricks.as_strided(audio, (frames, size), (audio.strides[0] * hop, audio.strides[0]))
    return np.abs(np.fft.rfft(view * window, axis=1))


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    reference = np.asarray(reference, np.float64).ravel()
    candidate = np.asarray(candidate, np.float64).ravel()
    if reference.shape != candidate.shape:
        return {"shape_mismatch": [int(reference.size), int(candidate.size)]}
    noise = ((reference - candidate) ** 2).sum()
    a, b = stft_magnitude(reference), stft_magnitude(candidate)
    floor = 1e-5 * max(a.max(), 1e-12)
    return {
        "log_spectral_distance_db": float(np.mean(np.abs(20 * np.log10((a + floor) / (b + floor))))),
        "spectral_convergence": float(np.linalg.norm(a - b) / np.linalg.norm(a)),
        "waveform_snr_db": float("inf") if noise == 0 else float(10 * np.log10((reference ** 2).sum() / noise)),
        "correlation": float(np.corrcoef(reference, candidate)[0, 1]),
    }


def save_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(np.clip(np.asarray(audio) * 32767, -32768, 32767).astype("<i2").tobytes())


def timed(function, repeats: int) -> tuple[object, float]:
    result = function()
    samples = []
    for _ in range(repeats):
        at = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - at)
    return result, statistics.median(samples)


def measure(engine, text: str, repeats: int = 2) -> dict:
    """Whole-passage synthesis, plus what streaming changes about the first audio."""
    from .server import split_for_streaming

    speak = lambda piece: engine.synthesise(piece, trim=False)
    audio, whole = timed(lambda: speak(text), repeats)
    pieces = split_for_streaming(text)
    _, first = timed(lambda: speak(pieces[0]), repeats)
    streamed_total = whole
    if len(pieces) > 1:
        _, streamed_total = timed(lambda: np.concatenate([speak(piece) for piece in pieces]), repeats)
    seconds = audio.size / RATE
    return {
        "audio_seconds": seconds,
        "synthesis_seconds": whole,
        "realtime_factor": whole / seconds,
        "pieces": len(pieces),
        "first_audio_seconds_whole": whole,          # without streaming you wait for everything
        "first_audio_seconds_streamed": first,       # with streaming, only the first clause
        "streamed_total_seconds": streamed_total,
        "audio": audio,
    }


def run(models: Path, corpus: Path, variants: list[str], repeats: int = 2,
        reference_variant: str = "upstream", audio_dir: Path | None = None,
        threads: int = 4) -> dict:
    from .server import Engine

    fixtures = json.loads(corpus.read_text())["fixtures"]
    engines = {}
    results: dict[str, list[dict]] = {}
    reference_audio: dict[str, np.ndarray] = {}

    for variant in dict.fromkeys([reference_variant, *variants]):
        print(f"\n=== {variant}")
        # One resident model at a time: three at once puts this machine under memory
        # pressure and the timings stop meaning anything.
        engines.clear()
        gc.collect()
        engine = Engine(models, variant, threads=threads)
        engines[variant] = engine
        rows = []
        for fixture in fixtures:
            record = measure(engine, fixture["text"], repeats)
            audio = record.pop("audio")
            if variant == reference_variant:
                reference_audio[fixture["id"]] = audio
            elif fixture["id"] in reference_audio:
                record.update(metrics(reference_audio[fixture["id"]], audio))
            if audio_dir:
                save_wav(Path(audio_dir) / f"{fixture['id']}-{variant}.wav", audio)
            row = {"id": fixture["id"], "class": fixture["class"], **record}
            rows.append(row)
            quality = (f" log-spectral {row['log_spectral_distance_db']:.2f} dB"
                       if "log_spectral_distance_db" in row else " (reference)")
            print(f"  {row['id']:20} {row['audio_seconds']:5.2f}s audio  "
                  f"synth {row['synthesis_seconds']:5.2f}s  rtf {row['realtime_factor']:.3f}  "
                  f"first audio {row['first_audio_seconds_streamed']:5.2f}s{quality}")
        results[variant] = rows

    summary = {}
    for variant, rows in results.items():
        entry = {
            "median_realtime_factor": statistics.median(r["realtime_factor"] for r in rows),
            "median_synthesis_seconds": statistics.median(r["synthesis_seconds"] for r in rows),
            "median_first_audio_streamed": statistics.median(r["first_audio_seconds_streamed"] for r in rows),
        }
        distances = [r["log_spectral_distance_db"] for r in rows if "log_spectral_distance_db" in r]
        if distances:
            entry["median_log_spectral_db"] = statistics.median(distances)
            entry["worst_log_spectral_db"] = max(distances)
        base = summary.get(reference_variant, {}).get("median_synthesis_seconds")
        if base:
            entry["speedup_vs_" + reference_variant] = base / entry["median_synthesis_seconds"]
        summary[variant] = entry
    return {"summary": summary, "results": results}


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="kokoro-pi validate",
                                     description="Measure quality and speed of the built variants")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, help="defaults to the packaged held-out set")
    parser.add_argument("--variants", nargs="+", default=["float", "int8"])
    parser.add_argument("--reference", default="upstream", help="variant everything is compared against")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--audio-dir", type=Path, help="write a WAV per fixture and variant")
    parser.add_argument("--out", type=Path, help="write the full report as JSON")
    args = parser.parse_args(argv)

    corpus = args.corpus or Path(__file__).resolve().parents[2] / "corpus/heldout.json"
    report = run(args.models, corpus, args.variants, args.repeats, args.reference,
                 args.audio_dir, args.threads)
    print("\n=== summary")
    print(json.dumps(report["summary"], indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
