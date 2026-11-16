# npu-opt: Out-of-Tree MLIR Compiler Passes for Spatial NPU Acceleration

An out-of-tree MLIR optimizer targeting scratchpad SRAM architectures (`memory_space = 1`) and asynchronous DMA engines. Lowers high-level `linalg.matmul` operations on off-chip DDR into tiled, double-buffered, software-pipelined async DMA schedules.

## Implemented Passes
- **`--npu-tile-matmul`**: Tiles `linalg.matmul` along the reduction dimension ($K$) via `TilingInterface` (`scf::tileUsingSCF`) and promotes tiled DDR slices into contiguous scratchpad SRAM buffers (`memory_space = 1`).
- **`--npu-ping-pong-dma`**: Promotes single-buffered SRAM allocations (`memref<MxNxT, 1>`) into double-buffered ping-pong arenas (`memref<2xMxNxT, 1>`) paired with hardware synchronization tag buffers, decomposing tiled `scf.for` loops into a **Prologue**, **Steady-State Loop** (overlapping `DMA(i+1)` with `Compute(i)`), and **Epilogue**.

## Build & Run

    mkdir -p build && cd build
    cmake -G Ninja .. \
      -DMLIR_DIR=$HOME/llvm-project/build/lib/cmake/mlir \
      -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm
    ninja npu-opt

    # Run end-to-end pipeline (Tiling -> SRAM Promotion -> Ping-Pong Async DMA)
    ./tools/npu-opt ../test/npu-opt/e2e-matmul-pipeline.mlir --npu-tile-matmul --npu-ping-pong-dma
