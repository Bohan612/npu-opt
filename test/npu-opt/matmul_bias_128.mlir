module {
  func.func @matmul_bias(%A: memref<128x128xf16>, 
                         %B: memref<128x128xf16>, 
                         %C: memref<128x128xf16>, 
                         %bias: memref<128xf16>) {
                         
    // 1. The Matrix Multiplication
    linalg.matmul ins(%A, %B : memref<128x128xf16>, memref<128x128xf16>)
                  outs(%C : memref<128x128xf16>)

    // 2. The Epilogue Bias Addition
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>, // C (input)
        affine_map<(d0, d1) -> (d1)>,     // bias (broadcasted across columns)
        affine_map<(d0, d1) -> (d0, d1)>  // C (output)
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
}
