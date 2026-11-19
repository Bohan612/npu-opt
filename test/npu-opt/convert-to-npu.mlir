// RUN: npu-opt %s --convert-to-npu | FileCheck %s
#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func.func @lower_dma_and_contract
// CHECK-NOT:     memref.dma_start
// CHECK-NOT:     memref.dma_wait
// CHECK-NOT:     vector.contract
// CHECK:         %[[TOK:.*]] = npu.dma_load %{{.*}} to %{{.*}} : memref<64x64xf16> to memref<64x64xf16, 1> -> !npu.token
// CHECK:         npu.dma_wait %[[TOK]] : !npu.token
// CHECK:         %[[RES:.*]] = npu.mma %{{.*}}, %{{.*}}, %{{.*}} : vector<16x16xf16>, vector<16x16xf16> -> vector<16x16xf16>
// CHECK:         return %[[RES]] : vector<16x16xf16>
func.func @lower_dma_and_contract(%ddr: memref<64x64xf16>, %sram: memref<64x64xf16, 1>, %a: vector<16x16xf16>, %b: vector<16x16xf16>, %c: vector<16x16xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %tag = memref.alloc() : memref<1xindex>
  memref.dma_start %ddr[%c0, %c0], %sram[%c0, %c0], %c4096, %tag[%c0] : memref<64x64xf16>, memref<64x64xf16, 1>, memref<1xindex>
  memref.dma_wait %tag[%c0], %c4096 : memref<1xindex>
  %res = vector.contract {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %a, %b, %c : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf16>
  return %res : vector<16x16xf16>
}
