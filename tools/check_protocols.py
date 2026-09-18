#!/usr/bin/env python3
"""Exercise every protocol a running service speaks, and say what broke.

    python3 tools/check_protocols.py --url http://127.0.0.1:8080 --wyoming 10200

Written against a *running* service on purpose. The interesting failures in a
speech server are not unit-testable: a wrong Content-Type that a browser
refuses, a Wyoming header that Home Assistant cannot parse, a 401 on the route a
monitor polls. Those only show up over a socket.
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import urllib.error
import urllib.request

PASS, FAIL = "  ok  ", " FAIL "
state = {"failed": 0}


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{PASS if condition else FAIL}] {name}{f' -- {detail}' if detail else ''}")
    if not condition:
        state["failed"] += 1


def request(url: str, body: dict | None = None, key: str | None = None,
            headers: dict | None = None) -> tuple[int, bytes, dict]:
    head = {"Content-Type": "application/json", **(headers or {})}
    if key:
        head["Authorization"] = f"Bearer {key}"
    payload = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=payload, headers=head,
                                 method="POST" if payload else "GET")
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers)


def wav_header(data: bytes) -> tuple[str, int, int]:
    """Riff tag, sample rate, channels -- so a WAV is checked, not assumed."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return "", 0, 0
    rate, = struct.unpack("<I", data[24:28])
    channels, = struct.unpack("<H", data[22:24])
    return "RIFF", rate, channels


def http_checks(base: str, key: str | None) -> None:
    print("\n-- native API --")
    status, body, _ = request(f"{base}/healthz")
    check("GET /healthz is open and minimal", status == 200 and json.loads(body) == {"ready": True},
          f"{status} {body[:80]!r}")

    status, body, _ = request(f"{base}/readyz", key=key)
    ready = json.loads(body) if status == 200 else {}
    check("GET /readyz reports the build", status == 200 and "variant" in ready,
          f"{status} {body[:120]!r}")
    print(f"         variant={ready.get('variant')} voices={ready.get('voices')} "
          f"lang={ready.get('lang')} threads={ready.get('threads')}")

    status, body, _ = request(f"{base}/v1/voices", key=key)
    voices = json.loads(body).get("voices", []) if status == 200 else []
    check("GET /v1/voices lists voices with languages", status == 200 and bool(voices),
          f"{len(voices)} voices")

    status, body, headers = request(f"{base}/v1/tts", {"text": "Protocol check.", "format": "wav"}, key)
    tag, rate, channels = wav_header(body)
    check("POST /v1/tts returns a real 24 kHz mono WAV",
          status == 200 and tag == "RIFF" and rate == 24000 and channels == 1,
          f"{status} {tag or 'not a wav'} rate={rate} ch={channels} {len(body)} bytes")

    status, body, headers = request(f"{base}/v1/tts", {"text": "Endianness check.", "format": "l16"}, key)
    check("POST /v1/tts honours l16 (big-endian)",
          status == 200 and headers.get("X-Audio-Format") == "s16be",
          f"{status} X-Audio-Format={headers.get('X-Audio-Format')}")

    status, body, _ = request(f"{base}/v1/tts", {"text": ""}, key)
    check("empty text is a 400", status == 400, str(status))

    if key:
        status, _, _ = request(f"{base}/v1/tts", {"text": "no key"})
        check("a missing key is a 401", status == 401, str(status))


def openai_checks(base: str, key: str | None) -> None:
    print("\n-- OpenAI compatibility --")
    status, body, _ = request(f"{base}/v1/models", key=key)
    ids = [model["id"] for model in json.loads(body).get("data", [])] if status == 200 else []
    check("GET /v1/models answers", status == 200 and "kokoro" in ids, f"{status} {ids}")

    # What Open WebUI sends: mp3 by default, an OpenAI voice name, a model name
    # this service has never heard of.
    status, body, headers = request(f"{base}/v1/audio/speech", {
        "model": "tts-1", "input": "The OpenAI endpoint works.", "voice": "nova",
    }, key)
    fallback = headers.get("X-Kokoro-Format-Fallback")
    kind = headers.get("Content-Type", "")
    check("POST /v1/audio/speech with defaults returns audio",
          status == 200 and len(body) > 1000,
          f"{status} {kind} {len(body)} bytes{' (fell back: ' + fallback + ')' if fallback else ''}")
    check("an OpenAI voice name maps to a Kokoro voice",
          headers.get("X-Kokoro-Voice", "").startswith("af_nova") or bool(fallback),
          f"X-Kokoro-Voice={headers.get('X-Kokoro-Voice')}")
    if not fallback:
        check("mp3 is really an mp3", body[:3] == b"ID3" or body[:2] in (b"\xff\xfb", b"\xff\xf3"),
              repr(body[:4]))

    status, body, headers = request(f"{base}/v1/audio/speech", {
        "input": "Explicit wav.", "response_format": "wav",
    }, key)
    tag, rate, _ = wav_header(body)
    check("response_format wav is a WAV", status == 200 and tag == "RIFF" and rate == 24000,
          f"{status} {tag or 'not a wav'} {len(body)} bytes")

    status, body, headers = request(f"{base}/v1/audio/speech", {
        "input": "Raw samples.", "response_format": "pcm",
    }, key)
    check("response_format pcm is headerless 16-bit audio",
          status == 200 and len(body) % 2 == 0 and body[:4] != b"RIFF",
          f"{status} {len(body)} bytes")

    status, _, headers = request(f"{base}/v1/audio/speech", {
        "input": "Too fast.", "speed": 4.0,
    }, key)
    check("a speed outside this model's range is clamped, not refused",
          status == 200 and headers.get("X-Kokoro-Speed-Clamped") == "2.0",
          f"{status} clamped={headers.get('X-Kokoro-Speed-Clamped')}")

    status, body, _ = request(f"{base}/v1/audio/speech", {"input": "x", "voice": "not-a-voice"}, key)
    check("an unknown voice is a 400 in OpenAI's error shape",
          status == 400 and "error" in json.loads(body), str(status))


def wyoming_event(stream, kind: str, data: dict | None = None) -> None:
    header: dict = {"type": kind}
    if data is not None:
        header["data"] = data
    stream.write(json.dumps(header).encode() + b"\n")
    stream.flush()


def wyoming_read(stream) -> tuple[str, dict, bytes] | None:
    line = stream.readline()
    if not line:
        return None
    header = json.loads(line)
    data = header.get("data") or {}
    if header.get("data_length"):
        data = {**data, **json.loads(stream.read(int(header["data_length"])))}
    payload = stream.read(int(header["payload_length"])) if header.get("payload_length") else b""
    return str(header["type"]), data, payload


def wyoming_checks(host: str, port: int) -> None:
    print("\n-- Wyoming (Home Assistant) --")
    try:
        connection = socket.create_connection((host, port), timeout=10)
    except OSError as error:
        check(f"connect to {host}:{port}", False, str(error))
        return
    with connection, connection.makefile("rwb") as stream:
        wyoming_event(stream, "describe")
        event = wyoming_read(stream)
        kind, data = (event[0], event[1]) if event else ("", {})
        programs = data.get("tts") or []
        voices = programs[0].get("voices", []) if programs else []
        check("describe is answered with info", kind == "info" and bool(programs), kind)
        check("info advertises voices with languages",
              bool(voices) and all("languages" in voice for voice in voices),
              f"{len(voices)} voices")
        check("info advertises streaming synthesis",
              programs and programs[0].get("supports_synthesize_streaming") is True,
              str(programs[0].get("supports_synthesize_streaming") if programs else None))

        name = voices[0]["name"] if voices else None
        wyoming_event(stream, "synthesize",
                      {"text": "Home Assistant can speak.", "voice": {"name": name}})
        kinds, audio, rate = [], 0, 0
        while (event := wyoming_read(stream)) is not None:
            kind, data, payload = event
            kinds.append(kind)
            if kind == "audio-start":
                rate = data.get("rate", 0)
            audio += len(payload)
            if kind == "audio-stop":
                break
        check("synthesize returns audio-start, chunks and audio-stop",
              kinds[:1] == ["audio-start"] and kinds[-1:] == ["audio-stop"]
              and kinds.count("audio-chunk") > 0,
              f"{kinds.count('audio-chunk')} chunks, {audio} bytes, rate={rate}")
        check("the audio is 24 kHz", rate == 24000, str(rate))

        # The streaming flow, which is how a sentence arrives from an assistant.
        wyoming_event(stream, "synthesize-start", {"voice": {"name": name}})
        for part in ("Streaming ", "text ", "arrives ", "in pieces."):
            wyoming_event(stream, "synthesize-chunk", {"text": part})
        wyoming_event(stream, "synthesize-stop")
        kinds, audio = [], 0
        while (event := wyoming_read(stream)) is not None:
            kind, _, payload = event
            kinds.append(kind)
            audio += len(payload)
            if kind == "synthesize-stopped":
                break
        check("the streaming flow speaks and then confirms",
              "audio-start" in kinds and "audio-stop" in kinds and kinds[-1] == "synthesize-stopped",
              f"{kinds.count('audio-chunk')} chunks, {audio} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check every protocol a kokoro-pi service speaks")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="base URL, no trailing path")
    parser.add_argument("--api-key")
    parser.add_argument("--wyoming", type=int, help="Wyoming port, if it is enabled")
    parser.add_argument("--wyoming-host", default="127.0.0.1")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    print(f"checking {base}")
    http_checks(base, args.api_key)
    openai_checks(base, args.api_key)
    if args.wyoming:
        wyoming_checks(args.wyoming_host, args.wyoming)

    print()
    if state["failed"]:
        print(f"{state['failed']} check(s) failed")
        sys.exit(1)
    print("everything answered as documented")


if __name__ == "__main__":
    main()
