# `npu-opt` — Dual-Architecture MLIR Compiler for GPGPU & Custom NPU

`npu-opt` is an out-of-tree MLIR compiler driver that lowers high-level `linalg` / `memref` matrix multiplication workloads (including fused epilogue operations such as bias addition) into two parallel hardware execution models: a standard SIMT GPGPU backend and a custom spatial NPU backend.

Maintaining these pipelines side-by-side allows the GPGPU pipeline to serve as an executable golden reference for numerical correctness and structural verification, while the NPU pipeline maps the exact same frontend math to custom silicon memory hierarchies, fused epilogues, and systolic / vector execution units.

## Dual-Architecture Pipelines

1. **GPGPU Pipeline (`--gpu-matmul-pipeline`)**: Lowers `linalg.matmul` to `gpu.launch`, maps matrix tiles to AMDGPU workgroup shared memory (`addrspace(3)`), and translates through the ROCDL dialect directly into optimized AMDGPU LLVM IR.
2. **NPU Pipeline (`--npu-compile-pipeline`)**: Lowers `linalg.matmul` and fused epilogue ops (such as `linalg.generic` bias addition) through manual 16x16x16 SRAM tiling and epilogue fusion, double-buffered ping-pong DMA scheduling, custom `npu` dialect lowering (`npu.vec_add`, `npu.mma`, DMA ops), and LLVM IR intrinsics (`@npu_dma_push`, `@npu_dma_wait`, `@npu_cube_mma_c_harness`, `@npu_vec_add_c_harness`).

## Step-by-Step Guide to Develop This Compiler

### 1. Define the Custom `npu` Dialect (`include/NPU/`, `lib/NPU/`)
* Define custom types (such as `!npu.token`) and operations (such as `npu.mma`, `npu.vec_add`, and DMA push/wait ops).
* Register the dialect and verify round-trip parsing and printing (`mlir-opt` parse -> print).

### 2. Fuse Epilogues & Tile SRAM Blocks (`ConvertMatmulToGPU.cpp`, `FuseEpilogue.cpp`, `TileMatmul.cpp`)
* **GPU**: Map loops to grid/block coordinates and lower allocations to `memref.alloca` in `#gpu.address_space<workgroup>`.
* **NPU**: Explicitly partition `M`, `N`, and `K` loops into 16x16x16 SRAM tiles using `scf.for` and `memref.subview` over `memref` buffers, accumulating across `K` into the 16x16 output tile and fusing the `1x16` bias epilogue (`linalg.generic`) immediately after the `K` reduction loop before writing back.

### 3. Schedule Double-Buffered Ping-Pong DMA (`PingPongDMAPipelining.cpp`)
* Allocate double-buffered (`[2 x ...]`) SRAM buffers for ping (`0`) and pong (`1`) slices.
* Issue async DMA pushes ahead of compute, wait on ping (`0`) before the first compute stage while pong (`1`) loads, and wait on pong (`1`) right before the second compute stage.

### 4. Vectorize & Lower Compute Tiles (`VectorizeMatmul.cpp`, `ConvertToNPUDialect.cpp`)
* Lower `linalg.generic` bias additions (`arith.addf` over 2 DPS inputs) directly into `npu.vec_add` on the tiled `memref` subviews.
* Vectorize eligible compute tile loops into 16x16 vector operations (`vector.transfer_read`, `vector.contract` / `vector.reduction`, and `vector.transfer_write`), keeping `vector.contract` intact when `lowerContract = false` or lowering outer products when `lowerContract = true`.
* Convert async DMA starts/waits and `vector.contract` ops into `npu` dialect ops with token synchronization while preserving each `memref.dma_wait` insertion point.

### 5. Lower to Standard & Custom LLVM IR (`ConvertNPUToLLVM.cpp`, `LowerDMAToLLVM.cpp`, `tools/npu-opt.cpp`)
* **GPU Pipeline**: Outlines GPU kernels, maps `gpu` ops to `rocdl`, and translates to standard LLVM IR.
* **NPU Pipeline**: Expand strided subviews (`ExpandStridedMetadata`), lower affine expressions (`LowerAffine`), convert 2D vector transfers via SCF (`ConvertVectorToSCF`), lower SCF/Vector/Arith/Func/CF to LLVM, and lower `npu` ops and tiled `linalg.matmul` ops to external LLVM calls (`@npu_dma_push`, `@npu_dma_wait`, `@npu_cube_mma_c_harness`, `@npu_vec_add_c_harness`) by unpacking `MemRefDescriptor` pointers and offsets (`LLVM::GEPOp`).
* Run `ReconcileUnrealizedCastsPass` at the end of the pipeline so no leftover `builtin.unrealized_conversion_cast` ops remain.

### 6. Build, Simulate, and Verify with `FileCheck` (`test/npu-opt/`)
* Configure and build out-of-tree against LLVM/MLIR using CMake and Ninja.
* Translate lowered LLVM-dialect MLIR via `mlir-translate --mlir-to-llvmir` and `llc -O3 -filetype=obj` into a `.o` object file, then link against a C simulator harness (`main.c`) implementing `npu_cube_mma_c_harness` and `npu_vec_add_c_harness`.
* Add `.mlir` regression tests with `// RUN:` and `CHECK` directives to verify dialect round-trips, individual passes, and the end-to-end pipeline.

## Build, Simulate & Run Tests

```bash
mkdir -p build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm
ninja

# Run the GPGPU Reference Pipeline
./tools/npu-opt ../test/npu-opt/matmul_128_gpu.mlir --gpu-matmul-pipeline

# Run the Custom NPU Pipeline with Fused Bias Epilogue
./tools/npu-opt --npu-compile-pipeline ../test/npu-opt/matmul_bias_128.mlir > compiled.mlir
/path/to/llvm-project/build/bin/mlir-translate --mlir-to-llvmir compiled.mlir > compiled.ll
/path/to/llvm-project/build/bin/llc -O3 -filetype=obj compiled.ll -o matmul_npu.o

# Link with your C-simulator harness and execute
gcc main.c matmul_npu.o -o npu_simulator
./npu_simulator
