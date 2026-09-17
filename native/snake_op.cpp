// Fused Kokoro Snake activation for ONNX Runtime's CPU execution provider.
//
// Snake is x + (1/alpha) * sin(alpha * x)^2, which the exported graph spells as five
// elementwise nodes. Each one reads and writes a whole tensor, and at 24 kHz those
// tensors are megabytes, so the chain is bound by memory traffic rather than maths.
// This computes it in a single pass. The arithmetic is unchanged.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"

#include "ops.h"

#include <cmath>
#include <stdexcept>

struct SnakeKernel {
  SnakeKernel(const OrtApi&, const OrtKernelInfo*) {}

  void Compute(OrtKernelContext* raw_context) {
    Ort::KernelContext context(raw_context);
    auto input = context.GetInput(0);
    auto alpha = context.GetInput(1);
    auto inverse = context.GetInput(2);
    auto shape = input.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3 || shape[0] < 1 || shape[1] < 1 || shape[2] < 1) {
      throw std::runtime_error("SnakeFused expects nonempty NCT");
    }
    auto channels = static_cast<size_t>(shape[1]);
    auto alpha_count = alpha.GetTensorTypeAndShapeInfo().GetElementCount();
    auto inverse_count = inverse.GetTensorTypeAndShapeInfo().GetElementCount();
    if ((alpha_count != 1 && alpha_count != channels) ||
        (inverse_count != 1 && inverse_count != channels)) {
      throw std::runtime_error("SnakeFused expects scalar or per-channel parameters");
    }
    auto output = context.GetOutput(0, shape);
    struct Work {
      const float *x, *alpha, *inverse;
      float* y;
      size_t channels, width, alpha_count, inverse_count;
    } work{input.GetTensorData<float>(), alpha.GetTensorData<float>(), inverse.GetTensorData<float>(),
           output.GetTensorMutableData<float>(), channels, static_cast<size_t>(shape[2]),
           alpha_count, inverse_count};
    context.ParallelFor([](void* data, size_t row) {
      const auto& w = *static_cast<Work*>(data);
      const float a = w.alpha[w.alpha_count == 1 ? 0 : row % w.channels];
      const float inverse = w.inverse[w.inverse_count == 1 ? 0 : row % w.channels];
      const float* x = w.x + row * w.width;
      float* y = w.y + row * w.width;
      for (size_t i = 0; i < w.width; ++i) {
        const float sine = std::sin(a * x[i]);
        y[i] = x[i] + inverse * sine * sine;
      }
    }, static_cast<size_t>(shape[0]) * channels, 0, &work);
  }
};

struct SnakeOp : Ort::CustomOpBase<SnakeOp, SnakeKernel> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new SnakeKernel(api, info);
  }
  const char* GetName() const { return "SnakeFused"; }
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
};

void AddSnakeOp(Ort::CustomOpDomain& domain) {
  static SnakeOp op;
  domain.Add(&op);
}
