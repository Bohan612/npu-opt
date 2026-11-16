// RUN: npu-opt %s --npu-tile-matmul --npu-ping-pong-dma | FileCheck %s

func.func @e2e_matmul(%ddr_a: memref<64x256xf16>,
                      %ddr_b: memref<256x64xf16>,
                      %sram_c: memref<64x64xf16, 1>) {
  linalg.matmul ins(%ddr_a, %ddr_b : memref<64x256xf16>, memref<256x64xf16>)
                outs(%sram_c : memref<64x64xf16, 1>)
  return
}

// CHECK-LABEL: func.func @e2e_matmul
// Double-buffered SRAM allocations (Ping & Pong) + DMA tag buffers
// CHECK:       memref.alloc() : memref<2x64x64xf16, 1 : i32>
// CHECK:       memref.alloc() : memref<2x1xi32, 1 : i32>
// CHECK:       memref.alloc() : memref<2x64x64xf16, 1 : i32>
// CHECK:       memref.alloc() : memref<2x1xi32, 1 : i32>
// Prologue DMA prefetch
// CHECK:       memref.dma_start
// CHECK:       memref.dma_start
// Steady-state software-pipelined loop
// CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index)
// CHECK:         memref.dma_start
// CHECK:         memref.dma_start
// CHECK:         memref.dma_wait
// CHECK:         memref.dma_wait
// CHECK:         linalg.matmul
// CHECK:         scf.yield
// Epilogue drain
// CHECK:       memref.dma_wait
// CHECK:       memref.dma_wait
// CHECK:       linalg.matmul
