// Both custom operators live in one shared library, registered under the "kokoro_pi"
// domain. ONNX Runtime rejects a duplicate domain, so keeping them together avoids
// asking the host application to register two libraries.
#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"

// x + (1/alpha) * sin(alpha * x)^2 in a single pass.
void AddSnakeOp(Ort::CustomOpDomain& domain);

// Quantise per input channel, accumulate int8 products in int32 with SDOT, dequantise
// per output channel straight to float.
void AddQuantConvOp(Ort::CustomOpDomain& domain);
