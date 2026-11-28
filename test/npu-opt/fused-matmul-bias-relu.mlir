// RUN: npu-opt %s --npu-fuse-epilogue --npu-ping-pong-dma | FileCheck %s
// RUN: npu-opt %s --npu-compile-pipeline | FileCheck %s --check-prefix=E2E

// CHECK-LABEL: func.func @matmul_bias_relu
// CHECK:       npu.dma_load_2d
// CHECK:       scf.for
// CHECK:         linalg.matmul
// CHECK:       linalg.generic
// CHECK:         arith.addf %{{.*}}, %{{.*}} : f16
// CHECK:       npu.vec_relu %{{.*}} : memref<16x16xf16, 1>
// CHECK:       npu.dma_store_2d

// E2E-DAG: llvm.func @npu_vec_add_c_harness(!llvm.ptr<1>, !llvm.ptr<1>)
// E2E-DAG: llvm.func @npu_vec_relu_c_harness(!llvm.ptr<1>)
// E2E-LABEL: llvm.func @matmul_bias_relu
// E2E-NOT:     npu.vec_relu
// E2E:         llvm.call @npu_vec_add_c_harness
// E2E:         llvm.call @npu_vec_relu_c_harness
// E2E:         llvm.call @npu_dma_store_2d
// E2E:         llvm.return

func.func @matmul_bias_relu(%A: memref<128x128xf16>,
                            %B: memref<128x128xf16>,
                            %C: memref<128x128xf16>,
                            %bias: memref<128xf16>) {
  %c0 = arith.constant 0.0 : f16
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

  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%C : memref<128x128xf16>)
    outs(%C : memref<128x128xf16>) {
  ^bb0(%val: f16, %out_val: f16):
    %relu = arith.maximumf %val, %c0 : f16
    linalg.yield %relu : f16
  }
  return
}
