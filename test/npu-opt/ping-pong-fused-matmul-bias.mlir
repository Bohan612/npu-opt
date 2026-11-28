// RUN: npu-opt %s --npu-fuse-epilogue --npu-ping-pong-dma | FileCheck %s

func.func @matmul_bias(%A: memref<128x128xf16>,
                       %B: memref<128x128xf16>,
                       %C: memref<128x128xf16>,
                       %bias: memref<128xf16>) {
  linalg.matmul ins(%A, %B : memref<128x128xf16>, memref<128x128xf16>)
                outs(%C : memref<128x128xf16>)

  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%C, %bias : memref<128x128xf16>, memref<128xf16>)
    outs(%C : memref<128x128xf16>) {
  ^bb0(%c_val: f16, %bias_val: f16, %out_val: f16):
    %sum = arith.addf %c_val, %bias_val : f16
    linalg.yield %sum : f16
  }
  return
}

// CHECK-LABEL: func.func @matmul_bias
// CHECK:       %[[DB_A:.*]] = memref.alloc() : memref<2x16x16xf16, 1>
// CHECK:       %[[DB_B:.*]] = memref.alloc() : memref<2x16x16xf16, 1>
// CHECK:       %[[ACC:.*]] = memref.alloc() : memref<16x16xf16, 1>
// CHECK:       %[[BIAS_SRAM:.*]] = memref.alloc() : memref<16xf16, 1>
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           npu.dma_load_2d
// CHECK:           npu.dma_load_2d
// CHECK:           npu.dma_load_2d
// CHECK:           npu.dma_wait
// CHECK:           npu.dma_wait
// CHECK:           scf.for
// CHECK:             npu.dma_load_2d
// CHECK:             npu.dma_load_2d
// CHECK:             linalg.matmul
// CHECK:             npu.dma_wait
// CHECK:             npu.dma_wait
// CHECK:             scf.yield
// CHECK:           linalg.matmul
// CHECK:           npu.dma_wait
// CHECK:           linalg.generic
// CHECK:             arith.addf %{{.*}}, %{{.*}} : f16
// CHECK:             linalg.yield
// CHECK:           npu.dma_store_2d
// CHECK:           npu.dma_wait

// RUN: npu-opt %s --npu-fuse-epilogue --npu-ping-pong-dma --convert-linalg-to-loops --npu-lower-dma-to-llvm --convert-npu-to-llvm --convert-scf-to-cf --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | FileCheck %s --check-prefix=LLVM

// LLVM-LABEL: llvm.func @matmul_bias
// LLVM-NOT:     npu.dma_load_2d
// LLVM-NOT:     npu.dma_store_2d
// LLVM-NOT:     npu.dma_wait
// LLVM-NOT:     linalg.matmul
// LLVM-NOT:     linalg.generic
// LLVM:         llvm.fmul
// LLVM:         llvm.fadd
// LLVM:         llvm.return
