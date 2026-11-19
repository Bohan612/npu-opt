// RUN: npu-opt %s | FileCheck %s

// CHECK-LABEL: func.func @npu_intrinsics_roundtrip
// CHECK:         %[[TOK:.*]] = npu.dma_load %{{.*}} to %{{.*}} : memref<64x64xf16> to memref<64x64xf16, 1> -> !npu.token
// CHECK:         npu.dma_wait %[[TOK]] : !npu.token
// CHECK:         %[[OUT:.*]] = npu.mma %{{.*}}, %{{.*}}, %{{.*}} : vector<16x16xf16>, vector<16x16xf16> -> vector<16x16xf16>
// CHECK:         return %[[OUT]] : vector<16x16xf16>
func.func @npu_intrinsics_roundtrip(
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
