# Configuration

Four ways to say the same thing, in this order of precedence:

```
command line  >  environment  >  config file  >  default
```

A flag suits trying something out. An environment variable suits a container or
a systemd drop-in. A file suits a machine that should still be configured the
same way after a reinstall, and is the only one of the four you can read back in
six months and understand.

To see what is actually in force, and where each value came from:

```bash
kokoro-pi config
```

```
config file: /home/you/.config/kokoro-pi/config.toml
  searched: /home/you/kokoro-pi.toml, /home/you/.config/kokoro-pi/config.toml

  models          /home/you/.kokoro-pi/models    config file (…/config.toml)
  port            8080                           config file (…/config.toml)
  threads         2                              command line
  voice           bf_emma                        environment (KOKORO_PI_VOICE)
  api_key         set (24 characters)            config file (…/config.toml)
```

## The file

`install.sh` writes one to `~/.config/kokoro-pi/config.toml` and points the
service at it, so changing a port does not mean editing a systemd unit. It is
TOML. Top-level keys apply to every command; a `[serve]` table overrides them
for the service alone.

```toml
models = "/home/you/.kokoro-pi/models"
host   = "127.0.0.1"
port   = 8080
voice  = "af_heart"
threads = 4

[serve]
voices = ["af_heart", "bf_emma", "am_michael"]
api_key = "a-long-random-string"
```

Searched in order: `--config <path>`, `$KOKORO_PI_CONFIG`, `./kokoro-pi.toml`,
`$XDG_CONFIG_HOME/kokoro-pi/config.toml` (`~/.config/kokoro-pi/config.toml`).
The first one that exists is the one that is read — they do not merge. An
unknown key is a startup error naming the file, not a setting that silently does
nothing.

Restart after editing: `systemctl --user restart kokoro-pi`.

## Everything

| Setting | File key / env | Default | Meaning |
| --- | --- | --- | --- |
| `--models` | `models` / `KOKORO_PI_MODELS` | `—` | directory holding models.json and the built models |
| `--variant` | `variant` / `KOKORO_PI_VARIANT` | `—` | which built model to serve: int8, float or upstream |
| `--voice` | `voice` / `KOKORO_PI_VOICE` | `—` | the voice used when a request does not name one |
| `--voices` | `voices` / `KOKORO_PI_VOICES` | `—` | restrict the voices this service will speak; empty means every voice in the pack |
| `--lang` | `lang` / `KOKORO_PI_LANG` | `auto` | phonemiser language, or auto to take it from the voice's name prefix |
| `--speed` | `speed` / `KOKORO_PI_SPEED` | `1.0` | speaking rate used when a request does not name one |
| `--host` | `host` / `KOKORO_PI_HOST` | `127.0.0.1` | address to bind; the default is loopback only |
| `--port` | `port` / `KOKORO_PI_PORT` | `8080` | port to bind |
| `--threads` | `threads` / `KOKORO_PI_THREADS` | `4` | ONNX Runtime intra-op threads; match your core count |
| `--max-chars` | `max_chars` / `KOKORO_PI_MAX_CHARS` | `4000` | longest request accepted, in characters |
| `--wait-seconds` | `wait_seconds` / `KOKORO_PI_WAIT_SECONDS` | `30.0` | how long a request waits for the engine before 429; 0 fails fast |
| `--default-format` | `default_format` / `KOKORO_PI_DEFAULT_FORMAT` | `s16le` | format used when a request does not name one; l16 is big-endian |
| `--stream` | `stream` / `KOKORO_PI_STREAM` | `true` | stream clause by clause unless a request says otherwise |
| `--stream-limit` | `stream_limit` / `KOKORO_PI_STREAM_LIMIT` | `180` | longest streamed piece, in characters; smaller starts sooner and phrases worse |
| `--api-key` | `api_key` / `KOKORO_PI_API_KEY` | `—` | require this key on /v1 routes, as Bearer or X-API-Key; unset means no auth |
| `--allow-origin` | `allow_origin` / `KOKORO_PI_ALLOW_ORIGIN` | `—` | value for Access-Control-Allow-Origin, so a browser page can call this |
| `--verify` | `verify` / `KOKORO_PI_VERIFY` | `true` | verify every asset's checksum at startup |
| `--log-requests` | `log_requests` / `KOKORO_PI_LOG_REQUESTS` | `false` | log one line per request. Never includes the text being spoken |

Booleans take `1/true/yes/on` or `0/false/no/off` from the environment, and
`--stream` / `--no-stream` on the command line. Lists take a TOML array, or a
comma-separated string from the environment: `KOKORO_PI_VOICES=af_heart,bf_emma`.

## The ones worth thinking about

### `voices` — what this service will speak

The pack holds 54 voices. Narrowing that list makes `/v1/voices` an honest
menu for whatever is calling you, and turns a typo into a 400 rather than a
surprise accent. `voice` must be one of them, or startup fails saying so.

### `lang` — which phonemes the words become

Kokoro names voices `<language><gender>_<name>`, so the voice already says what
it was trained to speak. `lang = "auto"`, the default, reads it from there:

| Prefix | Language | Prefix | Language |
| --- | --- | --- | --- |
| `a` | `en-us` | `h` | `hi` |
| `b` | `en-gb` | `i` | `it` |
| `e` | `es` | `j` | `ja` |
| `f` | `fr-fr` | `p` | `pt-br` |
| | | `z` | `zh` |

This matters: before `auto`, `bf_emma` was read American phonemes, which is a
British voice pronouncing an American's vowels.

**Only `en-us` and `en-gb` are tested here.** The phonemiser is espeak-ng, and
for Japanese and Chinese, espeak's phonemes are not the ones Kokoro was trained
on — those voices exist in the pack and will produce audio, but it will be
wrong. Spanish, French, Hindi, Italian and Portuguese fare better without being
something this project claims. Set `lang` explicitly to read any voice in any
language on purpose.

### `stream` and `stream_limit` — latency against phrasing

Streaming sends each clause as soon as it exists, which is what makes speech
start in 1.6 s instead of 5.5 s. It costs about 8% more total time and changes
phrasing slightly, because the model sees clause boundaries where it would have
chosen its own pauses. `stream_limit` is where the line falls: smaller starts
sooner and phrases worse. Send `"stream": false` per request when you are
generating a file rather than talking to someone.

### `api_key` — the difference between a speech service and an open one

Unset, there is no authentication, which is why `host` defaults to loopback.
Set, `/v1/*` and `/readyz` require it:

```bash
curl -H "Authorization: Bearer $KEY" http://pi.local:8080/v1/tts -d '{"text":"Hello"}'
curl -H "X-API-Key: $KEY"            http://pi.local:8080/v1/voices
```

`/healthz` stays open and answers `{"ready": true}` and nothing else, so a
monitor does not need the key and a stranger learns nothing. The comparison is
constant-time. This is a shared secret over plain HTTP: it stops your flatmate,
not a network attacker. For anything beyond your own LAN, terminate TLS in front
of it.

### `allow_origin` — calling it from a web page

Set it and every response carries `Access-Control-Allow-Origin`, with `OPTIONS`
preflight answered. `"*"` means any page on the internet may use your Pi as a
speech service, which is fine on a loopback bind and a poor idea on a public
one.

### `threads`

Match your core count: 4 on a Pi 5. Scaling is not linear — 2 threads is 1.72×
one thread, 4 is 2.23× — so if you are serving several requests, two services
with 2 threads each will get more total work done than one with 4. See
[benchmarks](benchmarks.md).

### `verify`

Every asset's checksum is checked against the manifest at startup, so a truncated
download or a half-finished build fails loudly instead of producing quietly wrong
audio. `--no-verify` skips it, which is worth about a second of startup on a Pi 5
and is not worth having.

## Examples

**A second service on another port, sharing the same models** — useful for an
int8 and a float engine side by side:

```bash
kokoro-pi serve --models ~/.kokoro-pi/models --port 8081 --variant float
```

**Serve the house, with a key**, in `~/.config/kokoro-pi/config.toml`:

```toml
models  = "/home/you/.kokoro-pi/models"
host    = "0.0.0.0"
port    = 8080
api_key = "a-long-random-string"
voices  = ["af_heart", "bf_emma"]
```

**A big-endian client that fails fast** — what Volitive, the project this came
out of, actually runs:

```toml
default_format = "l16"
wait_seconds   = 0
```

**Docker, or anything else that configures through the environment:**

```bash
docker run --rm -p 8080:8080 \
  -e KOKORO_PI_MODELS=/models \
  -e KOKORO_PI_HOST=0.0.0.0 \
  -e KOKORO_PI_VOICES=af_heart,bf_emma \
  -e KOKORO_PI_API_KEY="$KEY" \
  -v ~/.kokoro-pi/models:/models:ro kokoro-pi
```

**A systemd drop-in**, when you want one setting changed and the rest left alone:

```ini
# ~/.config/systemd/user/kokoro-pi.service.d/override.conf
[Service]
Environment=KOKORO_PI_THREADS=2
```

## Request fields

Per-request, every one optional:

| Field | Meaning |
| --- | --- |
| `text` | what to say. Required |
| `voice` | any voice this service offers |
| `lang` | phonemiser language for this request |
| `speed` | 0.5 to 2.0 |
| `format` | `s16le`, `l16` (big-endian) or `wav` |
| `stream` | `false` to get one complete body instead of clauses |

```bash
curl -s http://127.0.0.1:8080/v1/tts -H 'Content-Type: application/json' \
  -d '{"text":"Mind the gap.","voice":"bf_emma","speed":0.95}' | aplay -q -r 24000 -f S16_LE -c 1 -
```

`GET /v1/tts?text=Hello&format=wav` takes the same fields as query parameters,
which is enough for an `<audio src>`.
