// RUN: npu-opt %s --npu-tile-matmul --npu-ping-pong-dma --convert-linalg-to-loops --npu-lower-dma-to-llvm --convert-scf-to-cf --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | FileCheck %s

func.func @e2e_matmul(%ddr_a: memref<64x256xf16>,
                      %ddr_b: memref<256x64xf16>,
                      %sram_c: memref<64x64xf16, 1>) {
  linalg.matmul ins(%ddr_a, %ddr_b : memref<64x256xf16>, memref<256x64xf16>)
                outs(%sram_c : memref<64x64xf16, 1>)
  return
}

// CHECK-DAG: llvm.func @npu_dma_start(!llvm.ptr, !llvm.ptr<1>, i64, !llvm.ptr<1>)
// CHECK-DAG: llvm.func @npu_dma_wait(!llvm.ptr<1>, i64)
// CHECK-LABEL: llvm.func @e2e_matmul
// CHECK:       llvm.call @npu_dma_start
// CHECK:       llvm.call @npu_dma_start
// CHECK:       llvm.call @npu_dma_wait
// CHECK:       llvm.call @npu_dma_wait
// CHECK:       llvm.fmul
// CHECK:       llvm.fadd
