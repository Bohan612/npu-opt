// RUN: npu-opt %s --npu-compile-pipeline | FileCheck %s

// CHECK-DAG: llvm.func @npu_vec_add_c_harness(!llvm.ptr<1>, !llvm.ptr<1>)
// CHECK-DAG: llvm.func @npu_dma_load_2d(!llvm.ptr, !llvm.ptr, i64, i64, i64) -> i64
// CHECK-DAG: llvm.func @npu_dma_wait(i64)
// CHECK-DAG: llvm.func @npu_dma_store_2d(!llvm.ptr, !llvm.ptr, i64, i64, i64) -> i64

// CHECK-LABEL: llvm.func @matmul_bias
// CHECK-NOT:     linalg.matmul
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     npu.dma_load
// CHECK-NOT:     npu.dma_wait
// CHECK-NOT:     npu.mma
// CHECK-NOT:     builtin.unrealized_conversion_cast
// CHECK:         llvm.call @npu_dma_load_2d
// CHECK:         llvm.call @npu_dma_wait
// CHECK:         llvm.call @npu_vec_add_c_harness
// CHECK:         llvm.call @npu_dma_store_2d
// CHECK:         llvm.return
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
