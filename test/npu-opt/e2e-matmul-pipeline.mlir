// RUN: npu-opt %s --npu-compile-pipeline | FileCheck %s

// CHECK-DAG: llvm.func @npu_cube_mma(!llvm.array<16 x vector<16xf16>>, !llvm.array<16 x vector<16xf16>>, !llvm.array<16 x vector<16xf16>>) -> !llvm.array<16 x vector<16xf16>>
// CHECK-DAG: llvm.func @npu_dma_wait(i64)
// CHECK-DAG: llvm.func @npu_dma_push(!llvm.ptr, !llvm.ptr<1>, i64) -> i64

// CHECK-LABEL: llvm.func @matmul_128x128x128_e2e
// CHECK-NOT:     linalg.matmul
// CHECK-NOT:     memref.dma_start
// CHECK-NOT:     vector.contract
// CHECK-NOT:     npu.dma_load
// CHECK-NOT:     npu.mma
// CHECK-NOT:     builtin.unrealized_conversion_cast
// CHECK:         llvm.call @npu_dma_push
// CHECK:         llvm.call @npu_dma_wait
// CHECK:         llvm.call @npu_cube_mma
// CHECK:         llvm.return
func.func @matmul_128x128x128_e2e(
    %A: memref<128x128xf16>,
    %B: memref<128x128xf16>,
    %C: memref<128x128xf16>) {
  linalg.matmul ins(%A, %B : memref<128x128xf16>, memref<128x128xf16>)
                outs(%C : memref<128x128xf16>)
  return
}
