// RUN: npu-opt %s --npu-compile-pipeline | FileCheck %s

// CHECK-DAG: llvm.func @npu_cube_mma_c_harness(!llvm.ptr, !llvm.ptr, !llvm.ptr)

// CHECK-LABEL: llvm.func @matmul_128x128x128_e2e
// CHECK-NOT:     linalg.matmul
// CHECK-NOT:     memref.dma_start
// CHECK-NOT:     vector.contract
// CHECK-NOT:     npu.dma_load
// CHECK-NOT:     npu.mma
// CHECK-NOT:     builtin.unrealized_conversion_cast
// CHECK:         llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:         llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:         llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:         llvm.call @npu_cube_mma_c_harness(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:         llvm.return
func.func @matmul_128x128x128_e2e(
    %A: memref<128x128xf16>,
    %B: memref<128x128xf16>,
    %C: memref<128x128xf16>) {
  linalg.matmul ins(%A, %B : memref<128x128xf16>, memref<128x128xf16>)
                outs(%C : memref<128x128xf16>)
  return
}
