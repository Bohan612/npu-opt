#!/usr/bin/env bash
set -euo pipefail
IN_MLIR="${1:-test/npu-opt/matmul_128_gpu.mlir}"
OPT_BIN="./build/bin/npu-opt"
[[ -x "$OPT_BIN" ]] || OPT_BIN="./build/tools/npu-opt"

echo "[1/4] Running npu-opt --gpu-matmul-pipeline..."
"$OPT_BIN" "$IN_MLIR" --gpu-matmul-pipeline -o /tmp/gpu_lowered.mlir

echo "[2/4] Extracting GPU kernel to standalone module..."
python3 -c "
txt = open('/tmp/gpu_lowered.mlir').read()
out = []
in_gpu = False
for line in txt.splitlines():
    if line.strip().startswith('gpu.module '):
        in_gpu = True
        out.append('module {')
        continue
    if in_gpu:
        if line == '  }':
            out.append('}')
            break
        out.append(line)
open('/tmp/gpu_kernel_only.mlir', 'w').write('\n'.join(out))
"

echo "[3/4] Translating to LLVM IR..."
/home/bohanl/llvm-project/build/bin/mlir-translate --mlir-to-llvmir /tmp/gpu_kernel_only.mlir -o /tmp/gpu_kernel.ll

echo "[4/4] Optimizing LLVM IR..."
/home/bohanl/llvm-project/build/bin/opt -O3 -S /tmp/gpu_kernel.ll -o /tmp/gpu_kernel_opt.ll

echo -e "\n=== Success! Optimized AMDGPU Kernel IR (/tmp/gpu_kernel_opt.ll) ==="
