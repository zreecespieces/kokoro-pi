"""Replace the vocoder's convolutions with the fused int8 operator.

The scheme, and why it is this one:

* Activations are quantised **per input channel**, because these activations are
  heavy-tailed - the peak is 25-50x the standard deviation - and one scale for the
  whole tensor loses far too much resolution.
* An int32 dot product needs a single activation scale across its reduction, so the
  per-channel scales are folded into the weights offline (`v = w * s`) and then
  quantised per output channel. The weights are packed for the kernel's SDOT loop.
* Each layer's output leaves the kernel as **float**. ONNX Runtime's QLinearConv is
  faster per layer but must emit int8, and re-quantising every layer's output is what
  makes a straightforward int8 build sound wrong.

Calibration ranges come from `corpus/english.json`; quality is judged on the disjoint
`corpus/heldout.json`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

CHANNEL_BLOCK = 4
OUTPUT_TILE = 4
DOMAIN = "kokoro_pi"
SCOPE = "/generator/"


def find_targets(model: onnx.ModelProto, families: list[str] | None = None) -> list[dict]:
    """Every generator convolution the kernel can express."""
    initializers = {i.name: i for i in model.graph.initializer}
    producers = {o: n for n in model.graph.node for o in n.output}
    targets = []
    for node in model.graph.node:
        if node.op_type != "Conv" or SCOPE not in node.name:
            continue
        family = node.name.split(SCOPE)[-1].split("/")[0]
        if families and family not in families:
            continue
        name = node.input[1]
        while name not in initializers and name in producers:
            name = producers[name].input[0]  # the height-one rewrite feeds weights through Unsqueeze
        if name not in initializers:
            continue
        weight = numpy_helper.to_array(initializers[name])
        weight = weight.reshape(weight.shape[0], weight.shape[1], -1)
        out_channels, in_channels, kernel = weight.shape
        if out_channels % OUTPUT_TILE or in_channels % CHANNEL_BLOCK:
            continue  # conv_post has 22 output channels and stays in float
        attributes = {a.name: a for a in node.attribute}
        dilations = list(attributes["dilations"].ints) if "dilations" in attributes else [1, 1]
        pads = list(attributes["pads"].ints) if "pads" in attributes else [0, 0, 0, 0]
        strides = list(attributes["strides"].ints) if "strides" in attributes else [1, 1]
        if (attributes["group"].i if "group" in attributes else 1) != 1 or any(s != 1 for s in strides):
            continue
        if len(pads) == 4 and not pads[0] and not pads[2] and pads[1] == pads[3]:
            pad = pads[1]
        elif len(pads) == 2 and pads[0] == pads[1]:
            pad = pads[0]
        else:
            continue
        dilation = dilations[-1]
        if pad != dilation * (kernel - 1) // 2:
            continue  # the kernel implements "same" padding only
        targets.append({"node": node.name, "family": family, "input": node.input[0],
                        "output": node.output[0], "weight": name,
                        "bias": node.input[2] if len(node.input) > 2 else None,
                        "shape": [out_channels, in_channels, kernel],
                        # The activation's rank, which instrumentation needs to
                        # name the axes it reduces over. Taken from the
                        # convolution's own spatial rank rather than from the
                        # weight: the height-one rewrite leaves a 3-D weight
                        # initializer feeding a 2-D convolution through an
                        # Unsqueeze, so the weight says 3 where the activation
                        # is 4, and reducing the wrong axes yields a per-channel
                        # peak that is not per channel.
                        "rank": len(dilations) + 2,
                        "dilation": int(dilation), "pad": int(pad)})
    return targets


#: Suffix for the per-channel peak this adds to the graph.
PEAK = "__kokoro_peak"


def instrument(model: onnx.ModelProto, targets: list[dict]) -> onnx.ModelProto:
    """Add a per-input-channel peak for every target, and output *that*.

    The obvious instrumentation -- mark each target's input tensor as a graph
    output -- makes ONNX Runtime materialise and copy out 36 activation tensors
    per pass. At 128 channels and 24,001 samples that is 12 MB each, about
    430 MB in one dictionary, and it is the reason a build gets killed on a Pi
    with anything else running.

    All calibration wants is `abs(x).max()` over every axis but the channel, so
    the reduction happens in the graph and what comes back is a vector per
    target: kilobytes instead of hundreds of megabytes, and identical numbers.
    """
    instrumented = onnx.ModelProto()
    instrumented.CopyFrom(model)
    existing = {o.name for o in instrumented.graph.output}
    # Reductions moved `axes` from an attribute to an input in opset 18, and this
    # model is newer than that.
    reduce_takes_axes_input = any(
        entry.domain in ("", "ai.onnx") and entry.version >= 18
        for entry in instrumented.opset_import)
    for target in targets:
        name = target["input"]
        peak = f"{name}{PEAK}"
        if peak in existing:
            continue
        rank = int(target.get("rank") or 4)
        axes = [i for i in range(rank) if i != 1]
        instrumented.graph.node.append(
            helper.make_node("Abs", [name], [f"{peak}_abs"], name=f"{peak}_abs"))
        if reduce_takes_axes_input:
            axes_name = f"{peak}_axes"
            instrumented.graph.initializer.append(
                numpy_helper.from_array(np.array(axes, np.int64), axes_name))
            node = helper.make_node("ReduceMax", [f"{peak}_abs", axes_name], [peak],
                                    name=peak, keepdims=0)
        else:
            node = helper.make_node("ReduceMax", [f"{peak}_abs"], [peak],
                                    name=peak, axes=axes, keepdims=0)
        instrumented.graph.node.append(node)
        # Rank one, length unknown until the channel count is: a shape of None
        # leaves the value info without a `shape` field at all, which ONNX
        # Runtime tolerates and onnx.checker rejects.
        instrumented.graph.output.append(
            helper.make_tensor_value_info(peak, TensorProto.FLOAT, [None]))
    return instrumented


def accumulate_ranges(outputs: dict, targets: list[dict], ranges: dict) -> dict:
    """Per-input-channel peak magnitude, taken over everything seen so far."""
    for target in targets:
        peak_name = f"{target['input']}{PEAK}"
        if peak_name in outputs:
            peak = np.asarray(outputs[peak_name], np.float32).reshape(-1)
        else:
            # A graph instrumented the old way, or a caller passing raw tensors.
            activation = np.asarray(outputs[target["input"]], np.float32)
            axes = tuple(i for i in range(activation.ndim) if i != 1)
            peak = np.abs(activation).max(axis=axes)
        previous = ranges.get(target["node"])
        ranges[target["node"]] = peak if previous is None else np.maximum(previous, peak)
    return ranges


def pack(weight: np.ndarray, activation_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fold the activation scales in, quantise per output channel, pack as [O/4, K, C/4, 4, 4]."""
    out_channels, in_channels, kernel = weight.shape
    folded = weight.astype(np.float64) * activation_scale.reshape(1, -1, 1)
    output_scale = np.abs(folded.reshape(out_channels, -1)).max(axis=1) / 127.0
    output_scale[output_scale == 0] = 1.0
    quantised = np.clip(np.rint(folded / output_scale.reshape(-1, 1, 1)), -128, 127).astype(np.int8)
    packed = quantised.reshape(out_channels // OUTPUT_TILE, OUTPUT_TILE,
                               in_channels // CHANNEL_BLOCK, CHANNEL_BLOCK, kernel)
    return packed.transpose(0, 4, 2, 1, 3).copy(), output_scale.astype(np.float32)


def build(model: onnx.ModelProto, targets: list[dict], ranges: dict) -> tuple[onnx.ModelProto, list[dict]]:
    quantised = onnx.ModelProto()
    quantised.CopyFrom(model)
    initializers = {i.name: i for i in quantised.graph.initializer}
    nodes = {n.name: n for n in quantised.graph.node}
    report = []
    for index, target in enumerate(targets):
        peak = np.asarray(ranges[target["node"]], np.float64)
        weight = numpy_helper.to_array(initializers[target["weight"]])
        weight = weight.reshape(weight.shape[0], weight.shape[1], -1)
        if peak.size != weight.shape[1]:
            raise SystemExit(f"{target['node']}: calibration covers {peak.size} channels, "
                             f"weights need {weight.shape[1]}")
        scale = np.maximum(peak, 1e-6) / 127.0
        packed, output_scale = pack(weight, scale)
        if target["bias"]:
            bias = numpy_helper.to_array(initializers[target["bias"]]).astype(np.float32).ravel()
        else:
            bias = np.zeros(weight.shape[0], dtype=np.float32)
        prefix = f"kokoropi_int8_{index}"
        for name, array in [(f"{prefix}_act_scale", scale.astype(np.float32)),
                            (f"{prefix}_weights", packed),
                            (f"{prefix}_out_scale", output_scale),
                            (f"{prefix}_bias", bias)]:
            quantised.graph.initializer.append(numpy_helper.from_array(array, name))
        nodes[target["node"]].CopyFrom(helper.make_node(
            "QuantConv2d",
            [target["input"], f"{prefix}_act_scale", f"{prefix}_weights",
             f"{prefix}_out_scale", f"{prefix}_bias"],
            [target["output"]], name=target["node"] + "_int8", domain=DOMAIN,
            dilation=target["dilation"], pad=target["pad"]))
        report.append({"node": target["node"], "family": target["family"], "shape": target["shape"],
                       "activation_peak": float(peak.max())})
    if DOMAIN not in {entry.domain for entry in quantised.opset_import}:
        quantised.opset_import.append(helper.make_opsetid(DOMAIN, 1))
    return quantised, report


def load_ranges(path: Path) -> dict:
    with np.load(path) as stored:
        return {name: stored[name] for name in stored.files}


def save_ranges(path: Path, ranges: dict) -> None:
    np.savez(path, **ranges)


def summarise(report: list[dict]) -> dict:
    families: dict[str, int] = {}
    for entry in report:
        families[entry["family"]] = families.get(entry["family"], 0) + 1
    return {"convolutions": len(report), "families": families,
            "weights": int(sum(np.prod(e["shape"]) for e in report))}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")
