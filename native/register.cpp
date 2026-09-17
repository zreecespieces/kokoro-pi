// Single entry point ONNX Runtime calls when the library is registered with
// SessionOptions.register_custom_ops_library().
#include "ops.h"

#include <exception>

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* base) {
  Ort::InitApi(base->GetApi(ORT_API_VERSION));
  try {
    static Ort::CustomOpDomain domain("kokoro_pi");
    static const bool added = [&] {
      AddSnakeOp(domain);
      AddQuantConvOp(domain);
      return true;
    }();
    (void)added;
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& error) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, error.what());
  }
}
