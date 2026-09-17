"""kokoro-pi command line: build the models, serve them, measure them, speak."""
from __future__ import annotations

import sys

COMMANDS = {
    "build": "derive the optimised models from the upstream export",
    "serve": "run the resident HTTP speech service",
    "validate": "measure quality and speed across the built variants",
    "say": "speak some text through a running service",
}


def usage(code: int = 0) -> None:
    print("usage: kokoro-pi <command> [options]\n")
    width = max(len(name) for name in COMMANDS)
    for name, description in COMMANDS.items():
        print(f"  {name:<{width}}  {description}")
    print("\nrun `kokoro-pi <command> --help` for a command's options")
    raise SystemExit(code)


def say(argv: list[str]) -> None:
    import argparse
    import json
    import shutil
    import subprocess
    import urllib.error
    import urllib.request

    parser = argparse.ArgumentParser(prog="kokoro-pi say", description="Speak text through a running service")
    parser.add_argument("text", nargs="+")
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/tts")
    parser.add_argument("--voice")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--out", help="write a WAV here instead of playing it")
    args = parser.parse_args(argv)

    text = " ".join(args.text)
    body = {"text": text, "speed": args.speed, "format": "wav" if args.out else "s16le"}
    if args.voice:
        body["voice"] = args.voice
    request = urllib.request.Request(args.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.URLError as error:
        raise SystemExit(f"could not reach {args.url}: {error}\n"
                         "is the service running? `systemctl --user status kokoro-pi`")

    if args.out:
        with response, open(args.out, "wb") as output:
            shutil.copyfileobj(response, output)
        print(f"wrote {args.out}")
        return

    player = next((tool for tool in ("aplay", "ffplay", "play") if shutil.which(tool)), None)
    if not player:
        raise SystemExit("no aplay, ffplay or play on PATH; use --out to write a WAV instead")
    command = {
        "aplay": ["aplay", "-q", "-r", "24000", "-f", "S16_LE", "-c", "1", "-"],
        "ffplay": ["ffplay", "-hide_banner", "-loglevel", "error", "-nodisp", "-autoexit",
                   "-f", "s16le", "-ar", "24000", "-ac", "1", "-"],
        "play": ["play", "-q", "-t", "raw", "-r", "24000", "-e", "signed", "-b", "16", "-c", "1", "-"],
    }[player]
    # Stream straight into the player so speech starts before synthesis finishes.
    with response, subprocess.Popen(command, stdin=subprocess.PIPE) as sink:
        shutil.copyfileobj(response, sink.stdin)
        sink.stdin.close()
        sink.wait()


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        usage()
    command, argv = sys.argv[1], sys.argv[2:]
    if command == "build":
        from .build import main as run
        run(argv)
    elif command == "serve":
        from .server import main as run
        run(argv)
    elif command == "validate":
        from .validate import main as run
        run(argv)
    elif command == "say":
        say(argv)
    else:
        print(f"unknown command {command!r}\n")
        usage(2)


if __name__ == "__main__":
    main()
