"""Algebra-preserving rewrites of the exported Kokoro graph.

Four changes, none of which alter the mathematics:

1. Explicit per-channel temporal normalisation chains become InstanceNormalization.
   Kokoro is batch-one, so its [1,C,1] style scale and bias can be the [C] inputs of
   the fused operator without changing semantics.
2. One-dimensional float convolutions become equivalent height-one 2-D convolutions,
   which makes ONNX Runtime's blocked kernels eligible.
3. Squares spelled as Pow(x, 2) become Mul(x, x).
4. The five-node Snake activation chain becomes the single native SnakeFused operator.

Only the vocoder (the decoder's generator) is touched, because that is where 90% of the
time goes. The result is checked against the original waveform by `validate`.
"""
from __future__ import annotations

from collections import defaultdict
import copy

import numpy as np
import onnx
from onnx import helper, numpy_helper

SCOPE = "/decoder/decoder/generator/"
DOMAIN = "kokoro_pi"


def fuse_instance_norm(model: onnx.ModelProto) -> int:
    producers = {out: node for node in model.graph.node for out in node.output}
    consumers: dict[str, list[str]] = defaultdict(list)
    for node in model.graph.node:
        for value in node.input:
            consumers[value].append(node.name)
    constants = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    public_outputs = {value.name for value in model.graph.output}
    replacements: dict[str, list] = {}
    remove: set[str] = set()
    count = 0
    for final in model.graph.node:
        if not final.name.startswith(SCOPE) or final.op_type != "Add" or not final.name.endswith("/Add_3"):
            continue
        try:
            affine = producers[final.input[0]]
            divide = producers[affine.input[1]]
            centered = producers[divide.input[0]]
            mean = producers[centered.input[1]]
            sqrt = producers[divide.input[1]]
            epsilon = producers[sqrt.input[0]]
            variance = producers[epsilon.input[0]]
            square = producers[variance.input[0]]
            chain = [mean, centered, square, variance, epsilon, sqrt, divide, affine, final]
            if [n.op_type for n in chain] != ["ReduceMean", "Sub", "Mul", "ReduceMean", "Add",
                                              "Sqrt", "Div", "Mul", "Add"]:
                continue
            if list(square.input) != [centered.output[0], centered.output[0]] or mean.input[0] != centered.input[0]:
                continue
            if any(tuple(constants[n.input[1]].tolist()) not in [(-1,), (2,)] for n in [mean, variance]):
                continue
            names = {n.name for n in chain}
            if any((set(consumers[out]) - names) or out in public_outputs
                   for n in chain[:-1] for out in n.output):
                continue
            eps = float(constants[epsilon.input[1]].item())
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        prefix = f"kokoropi_norm_{count}"
        axes = prefix + "_axes"
        model.graph.initializer.append(numpy_helper.from_array(np.array([0, 2], dtype=np.int64), axes))
        replacements[final.name] = [
            helper.make_node("Squeeze", [affine.input[0], axes], [prefix + "_scale"], name=prefix + "_scale"),
            helper.make_node("Squeeze", [final.input[1], axes], [prefix + "_bias"], name=prefix + "_bias"),
            helper.make_node("InstanceNormalization",
                             [centered.input[0], prefix + "_scale", prefix + "_bias"],
                             list(final.output), name=final.name + "_fused", epsilon=eps),
        ]
        remove.update(names)
        count += 1
    _splice(model, replacements, remove)
    return count


def conv2d(model: onnx.ModelProto) -> int:
    nodes = []
    count = 0
    for node in model.graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type != "Conv" or not node.name.startswith(SCOPE) or len(attrs.get("kernel_shape", [])) != 1:
            nodes.append(node)
            continue
        prefix = f"kokoropi_conv_{count}"
        axes = prefix + "_axes"
        model.graph.initializer.append(numpy_helper.from_array(np.array([2], dtype=np.int64), axes))
        for key in ["kernel_shape", "dilations", "strides"]:
            if key in attrs:
                attrs[key] = [1, *attrs[key]]
        if "pads" in attrs:
            attrs["pads"] = [0, attrs["pads"][0], 0, attrs["pads"][1]]
        nodes.extend([
            helper.make_node("Unsqueeze", [node.input[0], axes], [prefix + "_x"], name=prefix + "_x"),
            helper.make_node("Unsqueeze", [node.input[1], axes], [prefix + "_w"], name=prefix + "_w"),
            helper.make_node("Conv", [prefix + "_x", prefix + "_w", *node.input[2:]], [prefix + "_y"],
                             name=node.name + "_2d", **attrs),
            helper.make_node("Squeeze", [prefix + "_y", axes], list(node.output), name=prefix + "_out"),
        ])
        count += 1
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return count


def square_mul(model: onnx.ModelProto) -> int:
    constants = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    count = 0
    for node in model.graph.node:
        if node.op_type == "Pow" and node.name.startswith(SCOPE) and node.input[1] in constants:
            exponent = constants[node.input[1]]
            if exponent.size == 1 and float(exponent.item()) == 2:
                node.op_type = "Mul"
                node.input[1] = node.input[0]
                count += 1
    return count


def fuse_snake(model: onnx.ModelProto) -> int:
    """Collapse the exact five-node Snake chain onto the native operator."""
    producers = {value: n for n in model.graph.node for value in n.output}
    consumers: dict[str, set[str]] = defaultdict(set)
    for node in model.graph.node:
        for value in node.input:
            consumers[value].add(node.name)
    replacements: dict[str, list] = {}
    remove: set[str] = set()
    for final in model.graph.node:
        if final.op_type != "Add" or not any(part in final.name for part in
                                             ["/generator/resblocks.", "/generator/noise_res."]):
            continue
        for x, product_name in [tuple(final.input), tuple(reversed(final.input))]:
            product = producers.get(product_name)
            if not product or product.op_type != "Mul":
                continue
            for square_name, inverse in [tuple(product.input), tuple(reversed(product.input))]:
                square = producers.get(square_name)
                if not square or square.op_type != "Mul" or square.input[0] != square.input[1]:
                    continue
                sine = producers.get(square.input[0])
                if not sine or sine.op_type != "Sin":
                    continue
                scaled = producers.get(sine.input[0])
                if not scaled or scaled.op_type != "Mul" or x not in scaled.input:
                    continue
                alpha = scaled.input[1] if scaled.input[0] == x else scaled.input[0]
                chain = [scaled, sine, square, product, final]
                names = {n.name for n in chain}
                if any(consumers[value] - names for n in chain[:-1] for value in n.output):
                    continue
                replacements[final.name] = [
                    helper.make_node("SnakeFused", [x, alpha, inverse], list(final.output),
                                     name=final.name + "_native", domain=DOMAIN)]
                remove.update(names)
    _splice(model, replacements, remove)
    if replacements and DOMAIN not in {entry.domain for entry in model.opset_import}:
        model.opset_import.append(helper.make_opsetid(DOMAIN, 1))
    return len(replacements)


def _splice(model: onnx.ModelProto, replacements: dict[str, list], remove: set[str]) -> None:
    nodes = []
    for node in model.graph.node:
        if node.name in replacements:
            nodes.extend(replacements[node.name])
        elif node.name not in remove:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def optimise(source: onnx.ModelProto) -> tuple[onnx.ModelProto, dict]:
    model = copy.deepcopy(source)
    counts = {
        "instance_norm_fused": fuse_instance_norm(model),
        "conv1d_as_conv2d": conv2d(model),
        "pow_to_mul": square_mul(model),
        "snake_fused": fuse_snake(model),
    }
    onnx.checker.check_model(model)
    return model, counts
