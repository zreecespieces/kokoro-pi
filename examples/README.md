# Examples

Every example assumes the service is running on `127.0.0.1:8080`. Set `KOKORO_PI_URL`
to point somewhere else.

| File | What it shows |
|---|---|
| [`say.sh`](say.sh) | Shortest useful thing: stream speech straight to the speaker with `aplay` |
| [`save_wav.sh`](save_wav.sh) | Save a WAV using the `GET` form, which is also browser friendly |
| [`client.py`](client.py) | Python streaming client that reports time to first audio |
| [`client.mjs`](client.mjs) | The same in Node 18+ |

## The one-liner worth remembering

```bash
curl -sS -N -X POST http://127.0.0.1:8080/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from my Raspberry Pi."}' \
  | aplay -q -r 24000 -f S16_LE -c 1 -
```

`-N` matters: it tells curl not to buffer, so playback starts on the first clause
instead of waiting for the whole passage.
