// Fused int8 convolution for ONNX Runtime's CPU execution provider.
//
// Why this exists: the vocoder's float convolutions already run at about 68% of the
// Pi's FP32 peak, so only lower precision can help. QLinearConv is fast but must
// write int8, and re-quantising every layer's output costs 4.8 dB of log-spectral
// distance. ConvInteger keeps int32 accumulation but pays for a 12 MB intermediate
// plus separate cast/scale/bias passes, reaching only 1.4-2.0x.
//
// This kernel does all three phases in one pass: quantise the input per channel,
// accumulate int8 products in int32 with SDOT, then dequantise per output channel
// straight to float. Nothing between the phases is materialised, and no output is
// ever rounded to int8.
//
// Per-input-channel activation scales cannot survive an int32 reduction, so they are
// folded into the weights offline (v = w * s, quantised per output channel). The
// weights therefore arrive pre-folded, pre-quantised and pre-packed.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"

#include "ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#if defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#define KOKORO_HAS_DOTPROD 1
#else
#define KOKORO_HAS_DOTPROD 0
#endif

namespace {

constexpr int64_t kChannelBlock = 4;   // SDOT consumes four int8 values per lane
constexpr int64_t kPositionTile = 16;  // four int32x4 accumulators per output channel
constexpr int64_t kOutputTile = 4;     // one weight load serves four output channels
constexpr int64_t kPositionBlock = 256;

struct Plan {
  const float* x;
  const float* activation_scale;
  const int8_t* weights;
  const float* output_scale;
  const float* bias;
  float* y;
  int8_t* packed;
  int64_t in_channels, out_channels, kernel, dilation, pad, width, padded_width, out_width, block;
};

// Quantise one group of four input channels and interleave them so that four
// consecutive positions of four channels form one contiguous 16-byte SDOT operand.
void PackChannelGroup(const Plan& plan, int64_t group) {
  const int64_t base = group * kChannelBlock;
  const int64_t stride = plan.padded_width * kChannelBlock;
  int8_t* destination = plan.packed + group * stride;
  std::fill(destination, destination + kChannelBlock * plan.pad, 0);
  std::fill(destination + (plan.pad + plan.width) * kChannelBlock,
            destination + stride, 0);

  float reciprocal[kChannelBlock];
  for (int64_t lane = 0; lane < kChannelBlock; ++lane) {
    const float scale = plan.activation_scale[base + lane];
    reciprocal[lane] = scale != 0.0f ? 1.0f / scale : 0.0f;
  }
  const float* rows[kChannelBlock];
  for (int64_t lane = 0; lane < kChannelBlock; ++lane) {
    rows[lane] = plan.x + (base + lane) * plan.width;
  }
  int8_t* out = destination + plan.pad * kChannelBlock;

  int64_t position = 0;
#if KOKORO_HAS_DOTPROD
  for (; position + 8 <= plan.width; position += 8) {
    int8x8_t lanes[kChannelBlock];
    for (int64_t lane = 0; lane < kChannelBlock; ++lane) {
      const float32x4_t low = vmulq_n_f32(vld1q_f32(rows[lane] + position), reciprocal[lane]);
      const float32x4_t high = vmulq_n_f32(vld1q_f32(rows[lane] + position + 4), reciprocal[lane]);
      // Round to nearest with ties to even, then saturate into int8.
      const int16x8_t narrowed = vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(low)),
                                              vqmovn_s32(vcvtnq_s32_f32(high)));
      lanes[lane] = vqmovn_s16(narrowed);
    }
    int8x8x4_t interleaved = {{lanes[0], lanes[1], lanes[2], lanes[3]}};
    vst4_s8(out + position * kChannelBlock, interleaved);
  }
#endif
  for (; position < plan.width; ++position) {
    for (int64_t lane = 0; lane < kChannelBlock; ++lane) {
      const float scaled = std::nearbyint(rows[lane][position] * reciprocal[lane]);
      out[position * kChannelBlock + lane] =
          static_cast<int8_t>(std::min(127.0f, std::max(-128.0f, scaled)));
    }
  }
}

// Accumulate one output tile straight into float, without an int8 or int32 round trip.
// One unit of work is a single output-channel tile over one block of positions, so
// both wide tensors and narrow deep ones have plenty of independent units.
void ComputeBlock(const Plan& plan, int64_t unit) {
  // Units that run next to each other share a position block, so a thread keeps that
  // block's activations hot while it streams the weights of successive tiles.
  const int64_t tiles = plan.out_channels / kOutputTile;
  const int64_t tile = unit % tiles;
  const int64_t block = unit / tiles;
  const int64_t start = block * plan.block;
  const int64_t stop = std::min(start + plan.block, plan.out_width);
  const int64_t groups = plan.in_channels / kChannelBlock;
  const int64_t weight_stride = groups * kChannelBlock * kChannelBlock;

  {
    const int8_t* tile_weights = plan.weights + tile * plan.kernel * weight_stride;
    float scales[kOutputTile], biases[kOutputTile];
    for (int64_t lane = 0; lane < kOutputTile; ++lane) {
      scales[lane] = plan.output_scale[tile * kOutputTile + lane];
      biases[lane] = plan.bias[tile * kOutputTile + lane];
    }
    int64_t position = start;
#if KOKORO_HAS_DOTPROD
    for (; position + kPositionTile <= stop; position += kPositionTile) {
      int32x4_t acc[kOutputTile][4];
      for (int64_t o = 0; o < kOutputTile; ++o) {
        for (int64_t p = 0; p < 4; ++p) acc[o][p] = vdupq_n_s32(0);
      }
      for (int64_t tap = 0; tap < plan.kernel; ++tap) {
        const int8_t* tap_weights = tile_weights + tap * weight_stride;
        const int64_t offset = position + tap * plan.dilation;
        for (int64_t group = 0; group < groups; ++group) {
          const int8x16_t w = vld1q_s8(tap_weights + group * 16);
          const int8_t* xp = plan.packed + (group * plan.padded_width + offset) * kChannelBlock;
          const int8x16_t x0 = vld1q_s8(xp);
          const int8x16_t x1 = vld1q_s8(xp + 16);
          const int8x16_t x2 = vld1q_s8(xp + 32);
          const int8x16_t x3 = vld1q_s8(xp + 48);
          acc[0][0] = vdotq_laneq_s32(acc[0][0], x0, w, 0);
          acc[0][1] = vdotq_laneq_s32(acc[0][1], x1, w, 0);
          acc[0][2] = vdotq_laneq_s32(acc[0][2], x2, w, 0);
          acc[0][3] = vdotq_laneq_s32(acc[0][3], x3, w, 0);
          acc[1][0] = vdotq_laneq_s32(acc[1][0], x0, w, 1);
          acc[1][1] = vdotq_laneq_s32(acc[1][1], x1, w, 1);
          acc[1][2] = vdotq_laneq_s32(acc[1][2], x2, w, 1);
          acc[1][3] = vdotq_laneq_s32(acc[1][3], x3, w, 1);
          acc[2][0] = vdotq_laneq_s32(acc[2][0], x0, w, 2);
          acc[2][1] = vdotq_laneq_s32(acc[2][1], x1, w, 2);
          acc[2][2] = vdotq_laneq_s32(acc[2][2], x2, w, 2);
          acc[2][3] = vdotq_laneq_s32(acc[2][3], x3, w, 2);
          acc[3][0] = vdotq_laneq_s32(acc[3][0], x0, w, 3);
          acc[3][1] = vdotq_laneq_s32(acc[3][1], x1, w, 3);
          acc[3][2] = vdotq_laneq_s32(acc[3][2], x2, w, 3);
          acc[3][3] = vdotq_laneq_s32(acc[3][3], x3, w, 3);
        }
      }
      for (int64_t o = 0; o < kOutputTile; ++o) {
        float* row = plan.y + (tile * kOutputTile + o) * plan.out_width + position;
        const float32x4_t scale = vdupq_n_f32(scales[o]);
        const float32x4_t bias = vdupq_n_f32(biases[o]);
        for (int64_t p = 0; p < 4; ++p) {
          vst1q_f32(row + p * 4, vfmaq_f32(bias, vcvtq_f32_s32(acc[o][p]), scale));
        }
      }
    }
#endif
    for (; position < stop; ++position) {
      for (int64_t o = 0; o < kOutputTile; ++o) {
        int32_t total = 0;
        for (int64_t tap = 0; tap < plan.kernel; ++tap) {
          const int8_t* tap_weights = tile_weights + tap * weight_stride;
          const int64_t offset = position + tap * plan.dilation;
          for (int64_t group = 0; group < groups; ++group) {
            const int8_t* xp = plan.packed + (group * plan.padded_width + offset) * kChannelBlock;
            const int8_t* wp = tap_weights + group * 16 + o * kChannelBlock;
            for (int64_t lane = 0; lane < kChannelBlock; ++lane) {
              total += static_cast<int32_t>(xp[lane]) * static_cast<int32_t>(wp[lane]);
            }
          }
        }
        plan.y[(tile * kOutputTile + o) * plan.out_width + position] =
            static_cast<float>(total) * scales[o] + biases[o];
      }
    }
  }
}

struct QuantConvKernel {
  QuantConvKernel(const OrtApi& api, const OrtKernelInfo* info) {
    Ort::ConstKernelInfo details(info);
    dilation_ = details.GetAttribute<int64_t>("dilation");
    pad_ = details.GetAttribute<int64_t>("pad");
    (void)api;
    if (dilation_ < 1 || pad_ < 0) throw std::runtime_error("QuantConv2d: bad dilation or pad");
  }

  void Compute(OrtKernelContext* raw_context) {
    Ort::KernelContext context(raw_context);
    auto input = context.GetInput(0);
    auto activation_scale = context.GetInput(1);
    auto weights = context.GetInput(2);
    auto output_scale = context.GetInput(3);
    auto bias = context.GetInput(4);

    auto shape = input.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() == 4) {
      if (shape[2] != 1) throw std::runtime_error("QuantConv2d expects a height-one image");
      shape = {shape[0], shape[1], shape[3]};
    }
    if (shape.size() != 3 || shape[0] != 1) throw std::runtime_error("QuantConv2d expects 1xCxW");

    auto packed_shape = weights.GetTensorTypeAndShapeInfo().GetShape();
    if (packed_shape.size() != 5 || packed_shape[3] != kOutputTile || packed_shape[4] != kChannelBlock) {
      throw std::runtime_error("QuantConv2d expects weights packed as [O/4, K, C/4, 4, 4]");
    }
    Plan plan{};
    plan.out_channels = packed_shape[0] * kOutputTile;
    plan.kernel = packed_shape[1];
    plan.in_channels = packed_shape[2] * kChannelBlock;
    plan.dilation = dilation_;
    plan.pad = pad_;
    plan.width = shape[2];
    plan.padded_width = plan.width + 2 * plan.pad;
    plan.out_width = plan.width + 2 * plan.pad - plan.dilation * (plan.kernel - 1);
    if (shape[1] != plan.in_channels) throw std::runtime_error("QuantConv2d: channel mismatch");
    if (plan.out_width < 1) throw std::runtime_error("QuantConv2d: padding too small for kernel");
    if (activation_scale.GetTensorTypeAndShapeInfo().GetElementCount() !=
            static_cast<size_t>(plan.in_channels) ||
        output_scale.GetTensorTypeAndShapeInfo().GetElementCount() !=
            static_cast<size_t>(plan.out_channels) ||
        bias.GetTensorTypeAndShapeInfo().GetElementCount() != static_cast<size_t>(plan.out_channels)) {
      throw std::runtime_error("QuantConv2d: scale or bias length mismatch");
    }

    std::vector<int64_t> output_shape{1, plan.out_channels, 1, plan.out_width};
    auto output = context.GetOutput(0, output_shape);
    plan.x = input.GetTensorData<float>();
    plan.activation_scale = activation_scale.GetTensorData<float>();
    plan.weights = weights.GetTensorData<int8_t>();
    plan.output_scale = output_scale.GetTensorData<float>();
    plan.bias = bias.GetTensorData<float>();
    plan.y = output.GetTensorMutableData<float>();

    buffer_.resize(static_cast<size_t>(plan.in_channels) * static_cast<size_t>(plan.padded_width));
    plan.packed = buffer_.data();

    context.ParallelFor([](void* data, size_t group) {
      PackChannelGroup(*static_cast<Plan*>(data), static_cast<int64_t>(group));
    }, static_cast<size_t>(plan.in_channels / kChannelBlock), 0, &plan);

    // Aim for many more units than cores; keep blocks a multiple of the position tile.
    int64_t block = ((plan.out_width / 16 + kPositionTile - 1) / kPositionTile) * kPositionTile;
    plan.block = std::min<int64_t>(std::max<int64_t>(block, 64), kPositionBlock);
    const int64_t blocks = (plan.out_width + plan.block - 1) / plan.block;
    const int64_t units = blocks * (plan.out_channels / kOutputTile);
    context.ParallelFor([](void* data, size_t unit) {
      ComputeBlock(*static_cast<Plan*>(data), static_cast<int64_t>(unit));
    }, static_cast<size_t>(units), 0, &plan);
  }

 private:
  int64_t dilation_ = 1;
  int64_t pad_ = 0;
  std::vector<int8_t> buffer_;
};

struct QuantConvOp : Ort::CustomOpBase<QuantConvOp, QuantConvKernel> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new QuantConvKernel(api, info);
  }
  const char* GetName() const { return "QuantConv2d"; }
  size_t GetInputTypeCount() const { return 5; }
  ONNXTensorElementDataType GetInputType(size_t index) const {
    return index == 2 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
};

}  // namespace

void AddQuantConvOp(Ort::CustomOpDomain& domain) {
  static QuantConvOp op;
  domain.Add(&op);
}
