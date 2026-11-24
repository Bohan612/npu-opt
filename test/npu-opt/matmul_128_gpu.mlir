func.func @matmul_128(%A: memref<128x128xf16>, %B: memref<128x128xf16>, %C: memref<128x128xf16>) {
  linalg.matmul ins(%A, %B : memref<128x128xf16>, memref<128x128xf16>)
                outs(%C : memref<128x128xf16>)
  return
}
