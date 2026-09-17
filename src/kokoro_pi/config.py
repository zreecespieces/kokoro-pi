"""Where settings come from, and the one order they come in.

    command line  >  environment  >  config file  >  default

Four ways to say the same thing, because the people running this are not all
running it the same way. A flag suits someone trying it out; an environment
variable suits a container or a systemd drop-in; a file suits a machine that
should still be configured the same way after a reinstall, and is the only one
of the four you can read back six months later and understand.

One table drives all of it -- argparse, the environment names, the file keys,
the validation and `kokoro-pi config` -- so an option cannot exist in one of
those and be missing from another.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib
from typing import Any, Callable, Iterable, Sequence

ENV_PREFIX = "KOKORO_PI_"

FORMATS = ("s16le", "l16", "wav")

#: Kokoro names voices `<language><gender>_<name>`, so the voice already says
#: which language it was trained to speak. `lang: auto` reads it from there,
#: which is how `bf_emma` gets British phonemes instead of American ones.
VOICE_LANGUAGES = {
    "a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr",
    "h": "hi", "i": "it", "j": "ja", "p": "pt-br", "z": "zh",
}

#: Languages the phonemiser handles well. The others are reachable -- nothing
#: stops you asking for them -- but espeak's phonemes for Japanese and Chinese
#: are not the ones Kokoro was trained on, so those voices sound wrong through
#: this path. See docs/configuration.md.
TESTED_LANGUAGES = ("en-us", "en-gb")


class ConfigError(SystemExit):
    """Wrong configuration is a startup failure, not an exception to catch."""


@dataclass(frozen=True)
class Option:
    name: str
    kind: str  # str | int | float | bool | path | list
    default: Any
    help: str
    choices: Sequence[str] | None = None
    #: Command line spelling, when it is not just `--<name-with-dashes>`.
    flags: Sequence[str] | None = None
    #: Not offered on the command line; file and environment only.
    file_only: bool = False

    @property
    def env(self) -> str:
        return f"{ENV_PREFIX}{self.name.upper()}"

    @property
    def cli(self) -> str:
        return f"--{self.name.replace('_', '-')}"


SERVE: tuple[Option, ...] = (
    Option("models", "path", None, "directory holding models.json and the built models"),
    Option("variant", "str", None, "which built model to serve: int8, float or upstream"),
    Option("voice", "str", None, "the voice used when a request does not name one"),
    Option("voices", "list", None,
           "restrict the voices this service will speak; empty means every voice in the pack"),
    Option("lang", "str", "auto",
           "phonemiser language, or auto to take it from the voice's name prefix"),
    Option("speed", "float", 1.0, "speaking rate used when a request does not name one"),
    Option("host", "str", "127.0.0.1", "address to bind; the default is loopback only"),
    Option("port", "int", 8080, "port to bind"),
    Option("threads", "int", 4, "ONNX Runtime intra-op threads; match your core count"),
    Option("max_chars", "int", 4000, "longest request accepted, in characters"),
    Option("wait_seconds", "float", 30.0,
           "how long a request waits for the engine before 429; 0 fails fast"),
    Option("default_format", "str", "s16le",
           "format used when a request does not name one; l16 is big-endian", choices=FORMATS),
    Option("stream", "bool", True, "stream clause by clause unless a request says otherwise"),
    Option("stream_limit", "int", 180,
           "longest streamed piece, in characters; smaller starts sooner and phrases worse"),
    Option("api_key", "str", None,
           "require this key on /v1 routes, as Bearer or X-API-Key; unset means no auth"),
    Option("allow_origin", "str", None,
           "value for Access-Control-Allow-Origin, so a browser page can call this"),
    Option("verify", "bool", True, "verify every asset's checksum at startup"),
    Option("log_requests", "bool", False,
           "log one line per request. Never includes the text being spoken"),
)


def _as_bool(value: Any, where: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{where}: expected true or false, got {value!r}")


def _as_list(value: Any, where: str) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.replace("\n", ",").split(",") if part.strip()]
    raise ConfigError(f"{where}: expected a list or a comma-separated string, got {value!r}")


def _coerce(option: Option, value: Any, where: str) -> Any:
    if value is None:
        return None
    try:
        if option.kind == "int":
            return int(value)
        if option.kind == "float":
            return float(value)
        if option.kind == "path":
            return Path(str(value)).expanduser()
        if option.kind == "bool":
            return _as_bool(value, where)
        if option.kind == "list":
            return _as_list(value, where)
        return str(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{where}: expected {option.kind}, got {value!r}") from None


class Settings:
    """Resolved configuration, and where each value came from."""

    def __init__(self, values: dict[str, Any], sources: dict[str, str]):
        self._values = values
        self.sources = sources

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(self, name: str) -> Any:
        return self._values[name]

    def items(self) -> Iterable[tuple[str, Any]]:
        return self._values.items()

    def redacted(self) -> dict[str, Any]:
        """The values, safe to print: a key is confirmed present, never shown."""
        out: dict[str, Any] = {}
        for name, value in self._values.items():
            if name == "api_key" and value:
                out[name] = f"set ({len(str(value))} characters)"
            elif isinstance(value, Path):
                out[name] = str(value)
            else:
                out[name] = value
        return out


def add_arguments(parser, options: Sequence[Option]) -> None:
    """Every option as a flag, defaulting to None so "not given" stays visible."""
    for option in options:
        if option.file_only:
            continue
        flags = list(option.flags or [option.cli])
        if option.kind == "bool":
            # --stream / --no-stream, so a file's `true` can be overridden on the
            # command line in both directions.
            parser.add_argument(*flags, dest=option.name, action="store_true",
                                default=None, help=option.help)
            parser.add_argument(f"--no-{option.name.replace('_', '-')}", dest=option.name,
                                action="store_false", default=None, help=f"do not {option.help}")
        elif option.kind == "list":
            parser.add_argument(*flags, dest=option.name, nargs="*", default=None, help=option.help)
        else:
            parser.add_argument(*flags, dest=option.name, default=None, help=option.help,
                                choices=list(option.choices) if option.choices else None)


def config_paths(explicit: str | Path | None = None) -> list[Path]:
    """Where a config file may live, most specific first.

    `./kokoro-pi.toml` is in the list on purpose: it makes a checkout you are
    experimenting in configure itself without touching anything in your home.
    """
    if explicit:
        return [Path(explicit).expanduser()]
    if os.environ.get(f"{ENV_PREFIX}CONFIG"):
        return [Path(os.environ[f"{ENV_PREFIX}CONFIG"]).expanduser()]
    xdg = os.environ.get("XDG_CONFIG_HOME")
    home = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return [Path.cwd() / "kokoro-pi.toml", home / "kokoro-pi" / "config.toml"]


def load_file(command: str, explicit: str | Path | None = None) -> tuple[dict[str, Any], Path | None]:
    """Read the first config file that exists.

    Top-level keys apply to every command; a `[serve]` or `[build]` table
    overrides them for that one, so one file can hold a machine's whole setup.
    """
    for path in config_paths(explicit):
        if not path.is_file():
            continue
        try:
            with path.open("rb") as source:
                document = tomllib.load(source)
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"{path}: {error}") from None
        values = {key: value for key, value in document.items() if not isinstance(value, dict)}
        section = document.get(command)
        if isinstance(section, dict):
            values.update(section)
        return values, path
    if explicit:
        raise ConfigError(f"no config file at {Path(explicit).expanduser()}")
    return {}, None


def resolve(options: Sequence[Option], args: Any = None, command: str = "serve",
            config: str | Path | None = None) -> Settings:
    """Merge the four sources into one set of values, and say where each came from."""
    given = {name: value for name, value in vars(args).items() if value is not None} if args else {}
    file_values, file_path = load_file(command, config or given.pop("config", None))
    known = {option.name for option in options}
    for key in file_values:
        if key not in known:
            raise ConfigError(f"{file_path}: unknown setting {key!r}\n"
                              f"  known settings: {', '.join(sorted(known))}")

    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for option in options:
        if option.name in given:
            values[option.name] = _coerce(option, given[option.name], option.cli)
            sources[option.name] = "command line"
        elif option.env in os.environ:
            values[option.name] = _coerce(option, os.environ[option.env], option.env)
            sources[option.name] = f"environment ({option.env})"
        elif option.name in file_values:
            values[option.name] = _coerce(option, file_values[option.name], str(file_path))
            sources[option.name] = f"config file ({file_path})"
        else:
            values[option.name] = option.default
            sources[option.name] = "default"
        chosen = values[option.name]
        if option.choices and chosen is not None and chosen not in option.choices:
            raise ConfigError(f"{option.name} must be one of {', '.join(option.choices)}; "
                              f"got {chosen!r} from {sources[option.name]}")
    return Settings(values, sources)


def language_for(voice: str, configured: str = "auto", default: str = "en-us") -> str:
    """The phonemiser language for a voice.

    `auto` reads it from the voice's name. Anything else is taken literally,
    because someone reading an American voice a British script is entitled to
    do that on purpose.
    """
    if configured and configured != "auto":
        return configured
    return VOICE_LANGUAGES.get(voice[:1], default)


def describe(settings: Settings, options: Sequence[Option]) -> str:
    """The effective configuration, with its provenance. `kokoro-pi config`."""
    redacted = settings.redacted()
    width = max(len(option.name) for option in options)
    lines = []
    for option in options:
        value = redacted[option.name]
        shown = "(unset)" if value is None else (",".join(value) if isinstance(value, list) else value)
        lines.append(f"  {option.name:<{width}}  {shown!s:<28}  {settings.sources[option.name]}")
    return "\n".join(lines)


def documentation(options: Sequence[Option]) -> str:
    """The options as a Markdown table, so the docs cannot drift from the code."""
    rows = ["| Setting | File key / env | Default | Meaning |", "| --- | --- | --- | --- |"]
    for option in options:
        default = "—" if option.default is None else (
            "true" if option.default is True else "false" if option.default is False
            else ",".join(option.default) if isinstance(option.default, list) else option.default)
        rows.append(f"| `{option.cli}` | `{option.name}` / `{option.env}` | `{default}` | {option.help} |")
    return "\n".join(rows)
