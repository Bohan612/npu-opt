# npu-opt: Out-of-Tree MLIR Compiler Passes for Spatial NPU Acceleration

An out-of-tree MLIR optimizer targeting scratchpad SRAM architectures (memory_space = 1) and asynchronous DMA engines. Implements automated double-buffering and software pipelining to hide off-chip DDR memory latency behind tensor compute.

## Build & Run

    mkdir build && cd build
    cmake -G Ninja .. \
      -DMLIR_DIR=$HOME/llvm-project/build/lib/cmake/mlir \
      -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm
    ninja npu-opt
    ./bin/npu-opt ../test/npu-opt/ping-pong-dma.mlir --npu-ping-pong-dma
