# `npu-opt`

Out-of-tree MLIR compiler pipeline (`linalg`/`memref` -> SRAM tiling -> ping-pong DMA -> 16x16 vectorization -> `npu` dialect -> LLVM lowering + `ReconcileUnrealizedCastsPass`).
