# `npu-opt` — Dual-Architecture MLIR Compiler for GPGPU & Custom NPU

`npu-opt` is an out-of-tree MLIR compiler driver that lowers high-level `linalg` / `memref` matrix multiplication workloads into two parallel hardware execution models: a standard SIMT GPGPU backend and a custom spatial NPU backend.

Maintaining these pipelines side-by-side allows the GPGPU pipeline to serve as an executable golden reference for numerical correctness and structural verification, while the NPU pipeline maps the exact same frontend math to custom silicon memory hierarchies and systolic arrays.

## Dual-Architecture Pipelines

1. **GPGPU Pipeline (`--gpu-matmul-pipeline`)**: Lowers `linalg.matmul` to `gpu.launch`, maps matrix tiles to AMDGPU workgroup shared memory (`addrspace(3)`), and translates through the ROCDL dialect directly into optimized AMDGPU LLVM IR.
2. **NPU Pipeline (`--npu-compile-pipeline`)**: Lowers `linalg.matmul` through SRAM tiling, double-buffered ping-pong DMA scheduling, 16x16 vectorization, custom `npu` dialect lowering, and LLVM IR intrinsics (`@npu_dma_push`, `@npu_dma_wait`, `@npu_cube_mma`).

## Step-by-Step Guide to Develop This Compiler

### 1. Define the Custom `npu` Dialect (`include/NPU/`, `lib/NPU/`)
* Define custom types (such as `!npu.token`) and operations (such as `npu.mma` and DMA push/wait ops).
* Register the dialect and verify round-trip parsing and printing (`mlir-opt` parse -> print).

### 2. Implement GPGPU and SRAM Tiling (`ConvertMatmulToGPU.cpp`, `TileMatmul.cpp`)
* **GPU**: Map loops to grid/block coordinates and lower allocations to `memref.alloca` in `#gpu.address_space<workgroup>`.
* **NPU**: Tile `linalg.matmul` / `memref` loops into SRAM-sized blocks (such as 16x16) to keep working sets in local memory (`memref` memory space `1`).

### 3. Schedule Double-Buffered Ping-Pong DMA (`PingPongDMAPipelining.cpp`)
* Allocate double-buffered (`[2 x ...]`) SRAM buffers for ping (`0`) and pong (`1`) slices.
* Issue async DMA pushes ahead of compute, wait on ping (`0`) before the first compute stage while pong (`1`) loads, and wait on pong (`1`) right before the second compute stage.

### 4. Vectorize Compute Tiles (`VectorizeMatmul.cpp`)
* Vectorize tile loops into 16x16 vector operations (`vector.transfer_read`, `vector.contract` / `vector.reduction`, and `vector.transfer_write`).
* Keep `vector.contract` intact when lowering through `--npu-compile-pipeline` (`lowerContract = false`), while supporting outer-product lowering when `lowerContract = true`.

### 5. Lower to the `npu` Dialect (`ConvertToNPUDialect.cpp`)
* Convert async DMA starts/waits and `vector.contract` ops into `npu` dialect ops with token synchronization.
* Preserve each `memref.dma_wait` insertion point when lowering so synchronization stays anchored at the consumer site.

### 6. Lower to Standard & Custom LLVM IR (`ConvertNPUToLLVM.cpp`, `LowerDMAToLLVM.cpp`, `tools/npu-opt.cpp`)
* **GPU Pipeline**: Outlines GPU kernels, maps `gpu` ops to `rocdl`, and translates to standard LLVM IR.
* **NPU Pipeline**: Expand strided subviews (`ExpandStridedMetadata`), lower affine expressions (`LowerAffine`), convert 2D vector transfers via SCF (`ConvertVectorToSCF`), lower SCF/Vector/Arith/Func/CF to LLVM, and lower `npu` ops to external LLVM calls (`@npu_dma_push`, `@npu_dma_wait`, `@npu_cube_mma`).
* Run `ReconcileUnrealizedCastsPass` at the end of the pipeline so no leftover `builtin.unrealized_conversion_cast` ops remain.

### 7. Build and Verify with `FileCheck` (`test/npu-opt/`)
* Configure and build out-of-tree against LLVM/MLIR using CMake and Ninja.
* Add `.mlir` regression tests with `// RUN:` and `CHECK` directives to verify dialect round-trips, individual passes, and the end-to-end pipeline.

## Build & Run Tests

```bash
mkdir -p build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm
ninja

# Run the GPGPU Reference Pipeline
./tools/npu-opt ../test/npu-opt/matmul_128_gpu.mlir --gpu-matmul-pipeline

# Run the Custom NPU Pipeline
./tools/npu-opt ../test/npu-opt/matmul_128_gpu.mlir --npu-compile-pipeline
```
