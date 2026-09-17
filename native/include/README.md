# Vendored ONNX Runtime headers

These are the public C and C++ API headers from
[ONNX Runtime](https://github.com/microsoft/onnxruntime) release 1.23, copied here
unmodified so `native/build.sh` works without a source checkout.

They declare **API version 23**. The custom operators call
`GetApi(ORT_API_VERSION)` at load time, so the runtime must implement at least that
version — which is why `requirements.txt` pins `onnxruntime==1.30.0`. An older runtime
will compile fine and then refuse to load the library.

ONNX Runtime is MIT licensed; see [../../NOTICE](../../NOTICE).
