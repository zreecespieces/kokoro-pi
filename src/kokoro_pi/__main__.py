"""kokoro-pi command line: build the models, serve them, measure them, speak."""
from __future__ import annotations

import sys

COMMANDS = {
    "build": "derive the optimised models from the upstream export",
    "serve": "run the resident HTTP speech service",
    "validate": "measure quality and speed across the built variants",
    "say": "speak some text through a running service",
    "config": "print the configuration in force, and where each value came from",
}


def usage(code: int = 0) -> None:
    print("usage: kokoro-pi <command> [options]\n")
    width = max(len(name) for name in COMMANDS)
    for name, description in COMMANDS.items():
        print(f"  {name:<{width}}  {description}")
    print("\nrun `kokoro-pi <command> --help` for a command's options")
    raise SystemExit(code)


def show_config(argv: list[str]) -> None:
    """What this machine is actually configured to do, and why."""
    import argparse

    from . import config as cfg

    parser = argparse.ArgumentParser(prog="kokoro-pi config",
                                     description="Print the configuration in force")
    parser.add_argument("--config", help="path to a TOML config file")
    parser.add_argument("--markdown", action="store_true", help="print the options as a table")
    args = parser.parse_args(argv)
    if args.markdown:
        print(cfg.documentation(cfg.SERVE))
        return
    _, path = cfg.load_file("serve", args.config)
    settings = cfg.resolve(cfg.SERVE, None, "serve", args.config)
    print(f"config file: {path or 'none found'}")
    print(f"  searched: {', '.join(str(candidate) for candidate in cfg.config_paths(args.config))}\n")
    print(cfg.describe(settings, cfg.SERVE))


def say(argv: list[str]) -> None:
    import argparse
    import json
    import shutil
    import subprocess
    import urllib.error
    import urllib.request

    from . import config as cfg

    parser = argparse.ArgumentParser(prog="kokoro-pi say", description="Speak text through a running service")
    parser.add_argument("text", nargs="+")
    parser.add_argument("--url", help="defaults to the host and port this machine is configured with")
    parser.add_argument("--voice")
    parser.add_argument("--lang", help="phonemiser language; the voice decides unless you say")
    parser.add_argument("--speed", type=float)
    parser.add_argument("--api-key", help="defaults to the configured key, if there is one")
    parser.add_argument("--config", help="path to a TOML config file")
    parser.add_argument("--out", help="write a WAV here instead of playing it")
    args = parser.parse_args(argv)

    # The same configuration the service reads, so `say` finds a service that
    # was moved to another port without being told about it twice.
    settings = cfg.resolve(cfg.SERVE, None, "serve", args.config)
    host = "127.0.0.1" if settings.host in ("0.0.0.0", "::") else settings.host
    url = args.url or f"http://{host}:{settings.port}/v1/tts"
    key = args.api_key or settings.api_key

    text = " ".join(args.text)
    body = {"text": text, "format": "wav" if args.out else "s16le"}
    if args.speed is not None:
        body["speed"] = args.speed
    if args.voice:
        body["voice"] = args.voice
    if args.lang:
        body["lang"] = args.lang
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:200]
        raise SystemExit(f"{url} answered {error.code}: {detail}")
    except urllib.error.URLError as error:
        raise SystemExit(f"could not reach {url}: {error}\n"
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
    elif command == "config":
        show_config(argv)
    else:
        print(f"unknown command {command!r}\n")
        usage(2)


if __name__ == "__main__":
    main()
