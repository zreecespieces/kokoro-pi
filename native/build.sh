#!/usr/bin/env bash
# Build the native operator library. Runs on the target machine - the int8 kernel uses
# ARM dot-product instructions, so it is compiled for the CPU it will run on.
set -euo pipefail
cd "$(dirname "$0")"

OUT=${1:-libkokoro_pi_ops.so}
CXX=${CXX:-g++}
ARCH_FLAGS=${ARCH_FLAGS:-}

if [ -z "$ARCH_FLAGS" ]; then
  case "$(uname -m)" in
    aarch64|arm64)
      # Cortex-A76 (Pi 5) and every Apple Silicon core support the dot-product extension.
      if grep -qi 'asimddp' /proc/cpuinfo 2>/dev/null || [ "$(uname -s)" = "Darwin" ]; then
        ARCH_FLAGS="-march=armv8.2-a+dotprod"
        [ "$(uname -s)" = "Darwin" ] && ARCH_FLAGS="-mcpu=apple-m1"
      else
        echo "warning: this ARM CPU does not report asimddp; int8 convolutions will use the" >&2
        echo "         portable fallback and will be slow. Expect no speedup from the int8 model." >&2
        ARCH_FLAGS="-march=armv8-a"
      fi
      ;;
    x86_64)
      echo "warning: x86_64 build. The int8 kernel has no AVX path, so it falls back to scalar" >&2
      echo "         code. Use the float model on this machine (see docs/troubleshooting.md)." >&2
      ARCH_FLAGS="-march=x86-64-v2"
      ;;
    *) ARCH_FLAGS="" ;;
  esac
fi

echo "building $OUT with $CXX $ARCH_FLAGS"
$CXX -std=c++17 -O3 $ARCH_FLAGS -ffp-contract=off -fPIC -shared \
  -Iinclude snake_op.cpp quant_conv_op.cpp register.cpp -o "$OUT"
case "$OUT" in /*) echo "built $OUT" ;; *) echo "built $(pwd)/$OUT" ;; esac
