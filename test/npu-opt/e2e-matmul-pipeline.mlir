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
// CHECK-DAG:     %[[C64:.*]] = llvm.mlir.constant(64 : i64) : i64
// CHECK-DAG:     %[[C8192:.*]] = llvm.mlir.constant(8192 : i64) : i64
// CHECK:         %[[SRAM_A:.*]] = llvm.addrspacecast %{{.*}} : !llvm.ptr to !llvm.ptr<1>
// CHECK:         %[[SRAM_B:.*]] = llvm.addrspacecast %{{.*}} : !llvm.ptr to !llvm.ptr<1>
// CHECK:         llvm.call @npu_dma_push(%{{.*}}, %[[SRAM_A]], %[[C8192]])
// CHECK:         llvm.call @npu_dma_push(%{{.*}}, %[[SRAM_B]], %[[C8192]])
// CHECK:         %[[A_PING_OFF:.*]] = llvm.add %{{.*}}, %[[C64]] : i64
// CHECK:         %[[B_PING_OFF:.*]] = llvm.add %{{.*}}, %[[C8192]] : i64
// CHECK:         %[[A_DDR_PING:.*]] = llvm.getelementptr %arg1[%[[A_PING_OFF]]] : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:         %[[A_SRAM_PING:.*]] = llvm.getelementptr %[[SRAM_A]][4096] : (!llvm.ptr<1>) -> !llvm.ptr<1>, f16
// CHECK:         llvm.call @npu_dma_push(%[[A_DDR_PING]], %[[A_SRAM_PING]], %[[C8192]])
// CHECK:         %[[B_DDR_PING:.*]] = llvm.getelementptr %arg8[%[[B_PING_OFF]]] : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK:         %[[B_SRAM_PING:.*]] = llvm.getelementptr %[[SRAM_B]][4096] : (!llvm.ptr<1>) -> !llvm.ptr<1>, f16
// CHECK:         llvm.call @npu_dma_push(%[[B_DDR_PING]], %[[B_SRAM_PING]], %[[C8192]])
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
