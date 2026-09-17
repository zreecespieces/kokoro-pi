// Streaming client for Node 18+. Prints when the first audio arrives.
//   node client.mjs "Hello from my Raspberry Pi."
const url = process.env.KOKORO_PI_URL ?? "http://127.0.0.1:8080/v1/tts";
const text = process.argv.slice(2).join(" ") || "Hello from my Raspberry Pi.";

const started = performance.now();
const response = await fetch(url, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text, format: "s16le" }),
});
if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);

let bytes = 0;
let first = null;
for await (const chunk of response.body) {
  first ??= performance.now() - started;
  bytes += chunk.length;
}
const seconds = bytes / 2 / 24000;
const total = performance.now() - started;
console.log(
  `backend ${response.headers.get("x-kokoro-backend")}: first audio after ${(first / 1000).toFixed(2)}s, ` +
  `${seconds.toFixed(2)}s of speech in ${(total / 1000).toFixed(2)}s`,
);
