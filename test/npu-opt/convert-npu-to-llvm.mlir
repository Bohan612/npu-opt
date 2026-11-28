// RUN: npu-opt %s --convert-npu-to-llvm --convert-func-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts | FileCheck %s

// CHECK-DAG: llvm.func @npu_cube_mma(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CHECK-DAG: llvm.func @npu_dma_wait(i64)
// CHECK-DAG: llvm.func @npu_dma_push(!llvm.ptr, !llvm.ptr, i64) -> i64
// CHECK-LABEL: llvm.func @npu_to_llvm_pipeline
// CHECK-NOT:     npu.dma_load
// CHECK-NOT:     npu.dma_wait
// CHECK-NOT:     npu.mma
// CHECK-NOT:     builtin.unrealized_conversion_cast
// CHECK:         llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:         llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
// CHECK:         %[[DST_PTR:.*]] = llvm.addrspacecast %{{.*}} : !llvm.ptr<1> to !llvm.ptr
// CHECK:         %[[BYTES:.*]] = llvm.mlir.constant(8192 : i64) : i64
// CHECK:         %[[TOK:.*]] = llvm.call @npu_dma_push(%{{.*}}, %[[DST_PTR]], %[[BYTES]]) : (!llvm.ptr, !llvm.ptr, i64) -> i64
// CHECK:         llvm.call @npu_dma_wait(%[[TOK]]) : (i64) -> ()
// CHECK:         llvm.call @npu_cube_mma(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
func.func @npu_to_llvm_pipeline(
    %ddr: memref<64x64xf16>,
    %sram: memref<64x64xf16, 1>,
    %a: vector<16x16xf16>,
    %b: vector<16x16xf16>,
    %c: vector<16x16xf16>) -> vector<16x16xf16> {
  %tok = npu.dma_load %ddr to %sram : memref<64x64xf16> to memref<64x64xf16, 1> -> !npu.token
  npu.dma_wait %tok : !npu.token
  %res = npu.mma %a, %b, %c : vector<16x16xf16>, vector<16x16xf16> -> vector<16x16xf16>
  return %res : vector<16x16xf16>
}
