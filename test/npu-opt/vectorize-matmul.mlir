// RUN: npu-opt %s --npu-vectorize-matmul | FileCheck %s

// CHECK-LABEL: func.func @sram_matmul_vectorize
// CHECK:         scf.for %{{.*}} = %c0 to %c64 step %c16
// CHECK:           scf.for %{{.*}} = %c0 to %c64 step %c16
// CHECK:             scf.for %{{.*}} = %c0 to %c64 step %c16
// CHECK:               vector.transfer_read
// CHECK:               vector.reduction <add>
// CHECK:             vector.transfer_write
// CHECK-NOT:     linalg.matmul
func.func @sram_matmul_vectorize(
    %A: memref<64x64xf16, 1>,
    %B: memref<64x64xf16, 1>,
    %C: memref<64x64xf16, 1>) {
  linalg.matmul ins(%A, %B : memref<64x64xf16, 1>, memref<64x64xf16, 1>)
                outs(%C : memref<64x64xf16, 1>)
  return
}
