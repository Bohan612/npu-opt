// RUN: npu-opt %s --npu-ping-pong-dma | FileCheck %s

func.func @matmul_sram_tiled(%ddr_a: memref<64x256xf16>,
                             %ddr_b: memref<256x64xf16>,
                             %sram_c: memref<64x64xf16, 1>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index

  %sram_a = memref.alloc() : memref<64x64xf16, 1>
  %sram_b = memref.alloc() : memref<64x64xf16, 1>

  scf.for %k = %c0 to %c256 step %c64 {
    %sub_a = memref.subview %ddr_a[0, %k] [64, 64] [1, 1]
        : memref<64x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
    %sub_b = memref.subview %ddr_b[%k, 0] [64, 64] [1, 1]
        : memref<256x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>

    memref.copy %sub_a, %sram_a
        : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16, 1>
    memref.copy %sub_b, %sram_b
        : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, 1>

    linalg.matmul ins(%sram_a, %sram_b : memref<64x64xf16, 1>, memref<64x64xf16, 1>)
                  outs(%sram_c : memref<64x64xf16, 1>)
  }
  return
}

// CHECK-LABEL: func.func @matmul_sram_tiled
// CHECK:       %[[DB_A:.*]] = memref.alloc() : memref<2x64x64xf16, 1>
// CHECK:       %[[TAG_A:.*]] = memref.alloc() : memref<2x1xi32, 1 : i32>
// CHECK:       %[[DB_B:.*]] = memref.alloc() : memref<2x64x64xf16, 1>
// CHECK:       %[[TAG_B:.*]] = memref.alloc() : memref<2x1xi32, 1 : i32>
// CHECK:       memref.dma_start
// CHECK:       memref.dma_start
// CHECK:       scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[STAGE:.*]] = %{{.*}}) -> (index)
// CHECK:         memref.dma_start
// CHECK:         memref.dma_start
// CHECK:         memref.dma_wait
// CHECK:         memref.dma_wait
// CHECK:         linalg.matmul
// CHECK:         scf.yield
// CHECK:       memref.dma_wait
// CHECK:       memref.dma_wait
// CHECK:       linalg.matmul
