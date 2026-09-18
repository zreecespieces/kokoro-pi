"""Turn the upstream Kokoro export into the optimised models this service serves.

    upstream  kokoro-v1.0.fp16.onnx   as published
    float     kokoro-fused.onnx       graph rewrites + the native Snake kernel
    int8      kokoro-int8.onnx        adds the fused int8 convolution kernel

Every stage prints what it did and how long it took, and both derived models are
checked against the upstream audio before they are written into the manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from . import assets, graph, paths, quantise, validate

UPSTREAM_MODEL = "kokoro-v1.0.fp16.onnx"
FLOAT_MODEL = "kokoro-fused.onnx"
INT8_MODEL = "kokoro-int8.onnx"
LIBRARY = paths.LIBRARY
PARITY_FLOOR_DB = 60.0


def session_for(model: Path, library: Path | None, threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    if library:
        options.register_custom_ops_library(str(library))
    return ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])


def capture_inputs(session, voices: Path, texts: list[str], voice: str):
    """Record the tensors Kokoro feeds the graph, so every variant can be run on identical input."""
    from kokoro_onnx import Kokoro

    kokoro = Kokoro.from_session(session, str(voices))
    captured: list[dict] = []
    original = session.run

    def capture(outputs, feeds, *extra):
        captured.append({key: np.array(value, dtype=np.int64 if key == "tokens" else np.float32, copy=True)
                         for key, value in feeds.items()})
        return original(outputs, feeds, *extra)

    session.run = capture
    collected = []
    for text in texts:
        captured.clear()
        kokoro.create(text, voice=voice, lang="en-us", trim=False)
        collected.append(list(captured))
    session.run = original
    return collected


def audio_for(session, feeds: list[dict]) -> np.ndarray:
    return np.concatenate([np.asarray(session.run(None, feed)[0]).ravel() for feed in feeds])


def stage(label: str):
    print(f"\n[{label}]", flush=True)
    return time.perf_counter()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kokoro-pi build",
                                     description="Build the optimised Kokoro models")
    parser.add_argument("--models", type=Path, required=True, help="where models and the manifest go")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--corpus", type=Path, default=None,
                        help="calibration texts; defaults to the packaged corpus/english.json")
    parser.add_argument("--heldout", type=Path, default=None,
                        help="texts the calibration never sees; defaults to corpus/heldout.json")
    parser.add_argument("--families", nargs="*", default=None,
                        help="resblock families to quantise; default is every eligible convolution")
    parser.add_argument("--skip-int8", action="store_true", help="build only the float model")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args(argv)

    import onnx

    models = args.models
    models.mkdir(parents=True, exist_ok=True)
    provenance: dict = {"built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "voice": args.voice, "threads": args.threads}

    at = stage("1/5 upstream assets")
    downloaded = assets.fetch_all(models)
    voices = downloaded["voices-v1.0.bin"]
    print(f"  done in {time.perf_counter() - at:.1f}s")

    at = stage("2/5 native operators")
    library = models / LIBRARY
    prebuilt = paths.prebuilt_library()
    if prebuilt:
        # A binary wheel carried one, compiled for this architecture in a
        # manylinux container. Nothing to do, and no compiler needed.
        shutil.copyfile(prebuilt, library)
        print(f"  using the operators this install shipped with ({prebuilt})")
    else:
        subprocess.run(["bash", str(paths.resource("native/build.sh")), str(library)], check=True)
    provenance["library"] = LIBRARY
    print(f"  done in {time.perf_counter() - at:.1f}s")

    at = stage("3/5 float graph: normalisation, height-one convolutions, squares, Snake")
    upstream = onnx.load(str(models / UPSTREAM_MODEL))
    fused, counts = graph.optimise(upstream)
    onnx.save(fused, str(models / FLOAT_MODEL))
    provenance["rewrites"] = counts
    print(f"  {json.dumps(counts)}")
    print(f"  done in {time.perf_counter() - at:.1f}s")

    corpus_path = args.corpus or paths.resource("corpus/english.json")
    heldout_path = args.heldout or paths.resource("corpus/heldout.json")
    corpus_texts = [entry["text"] for entry in json.loads(corpus_path.read_text())["fixtures"]]
    heldout = json.loads(heldout_path.read_text())["fixtures"]

    at = stage("4/5 float parity check")
    upstream_session = session_for(models / UPSTREAM_MODEL, None, args.threads)
    feeds = capture_inputs(upstream_session, voices, [entry["text"] for entry in heldout], args.voice)
    float_session = session_for(models / FLOAT_MODEL, library, args.threads)
    parity = []
    for entry, feed in zip(heldout, feeds):
        reference = audio_for(upstream_session, feed)
        candidate = audio_for(float_session, feed)
        measured = validate.metrics(reference, candidate)
        parity.append({"id": entry["id"], **measured})
        print(f"  {entry['id']:20} waveform SNR {measured['waveform_snr_db']:7.1f} dB")
    worst = min(row["waveform_snr_db"] for row in parity)
    if worst < PARITY_FLOOR_DB:
        raise SystemExit(f"float rewrites changed the waveform ({worst:.1f} dB < {PARITY_FLOOR_DB} dB); "
                         "refusing to write the manifest")
    provenance["float_parity"] = {"worst_waveform_snr_db": worst, "per_utterance": parity}
    print(f"  worst {worst:.1f} dB - these rewrites are algebra-preserving, so this should be high")
    print(f"  done in {time.perf_counter() - at:.1f}s")

    variants = {
        "upstream": {"model": UPSTREAM_MODEL, "custom_ops": [], "backend": "upstream-onnx"},
        "float": {"model": FLOAT_MODEL, "custom_ops": [LIBRARY], "backend": "float-fused"},
    }
    active = "float"

    if not args.skip_int8:
        at = stage("5/5 int8 convolutions: calibrate, build, check on held-out text")
        targets = quantise.find_targets(fused, args.families)
        print(f"  {json.dumps(quantise.summarise([{'family': t['family'], 'shape': t['shape']} for t in targets]))}")
        instrumented = quantise.instrument(fused, targets)
        instrumented_path = models / "kokoro-calibration.onnx"
        onnx.save(instrumented, str(instrumented_path))
        calibration_session = session_for(instrumented_path, library, args.threads)
        names = [o.name for o in calibration_session.get_outputs()]
        ranges: dict = {}
        calibration_feeds = capture_inputs(calibration_session, voices, corpus_texts, args.voice)
        for text, utterance in zip(corpus_texts, calibration_feeds):
            for feed in utterance:
                outputs = dict(zip(names, calibration_session.run(None, feed)))
                quantise.accumulate_ranges(outputs, targets, ranges)
            print(f"  calibrated on {len(text)} characters")
        del calibration_session
        instrumented_path.unlink()
        quantise.save_ranges(models / "calibration.npz", ranges)

        quantised, report = quantise.build(fused, targets, ranges)
        onnx.save(quantised, str(models / INT8_MODEL))
        provenance["int8"] = {"corpus": str(args.corpus.name), "utterances": len(corpus_texts),
                              **quantise.summarise(report)}
        variants["int8"] = {"model": INT8_MODEL, "custom_ops": [LIBRARY], "backend": "int8-fused"}
        active = "int8"

        int8_session = session_for(models / INT8_MODEL, library, args.threads)
        held = []
        for entry, feed in zip(heldout, feeds):
            reference = audio_for(upstream_session, feed)
            candidate = audio_for(int8_session, feed)
            measured = validate.metrics(reference, candidate)
            held.append({"id": entry["id"], **measured})
            print(f"  {entry['id']:20} log-spectral {measured['log_spectral_distance_db']:5.2f} dB")
        worst_spectral = max(row["log_spectral_distance_db"] for row in held)
        provenance["int8"]["heldout"] = {"worst_log_spectral_db": worst_spectral, "per_utterance": held}
        print(f"  worst {worst_spectral:.2f} dB against a 0.5-1.3 dB phase floor "
              f"(see docs/quality.md)")
        if worst_spectral > 4.0:
            print("  warning: that is high. Try --families to leave the sensitive groups in float, "
                  "or serve --variant float.")
        print(f"  done in {time.perf_counter() - at:.1f}s")

    checksums = {}
    for name in dict.fromkeys([UPSTREAM_MODEL, FLOAT_MODEL, "voices-v1.0.bin", LIBRARY,
                               *( [INT8_MODEL] if not args.skip_int8 else [] )]):
        path = models / name
        if path.exists():
            checksums[name] = assets.digest(path)

    manifest = {
        "kokoro_pi": 1,
        "voice": args.voice,
        "variant": active,
        "variants": variants,
        "checksums": checksums,
        "precision": ("upstream export is FLOAT with selected INT8 QLinear operators despite its "
                      "fp16 filename; the int8 variant additionally runs the vocoder's resblock "
                      "convolutions through the fused int8 kernel"),
        "build": provenance,
    }
    quantise.write_json(models / "models.json", manifest)

    print(f"\nmanifest written to {models / 'models.json'}")
    print(f"serving variant: {active}")
    if not args.skip_validation:
        print("\nrun `kokoro-pi validate --models %s` for the full speed and quality table" % models)


if __name__ == "__main__":
    main()
