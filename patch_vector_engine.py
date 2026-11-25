import os
import re

# 1. Patch NPUOps.td
td_file = 'include/NPU/NPUOps.td'
if os.path.exists(td_file):
    with open(td_file, 'r') as f:
        content = f.read()
    if 'NPU_VecAddOp' not in content:
        with open(td_file, 'a') as f:
            f.write("""
def NPU_VecAddOp : NPU_Op<"vec_add", []> {
  let summary = "NPU 16x16 vector-bias addition";
  let description = [{
    Executes a 16x16 vector addition in the NPU's Vector ALU. 
    Adds a 1x16 bias tile to a 16x16 matrix tile currently in SRAM.
  }];
  let arguments = (ins
    Arg<AnyMemRef, "the 16x16 SRAM matrix tile (in/out)", [MemWrite]>:$matrix_tile,
    Arg<AnyMemRef, "the 1x16 SRAM bias tile", [MemRead]>:$bias_tile
  );
  let assemblyFormat = "$matrix_tile `,` $bias_tile attr-dict `:` type($matrix_tile) `,` type($bias_tile)";
}
""")
        print(f"[SUCCESS] Injected NPU_VecAddOp into {td_file}")

# 2. Patch npu_baremetal_harness.c
c_file = 'npu_baremetal_harness.c'
if os.path.exists(c_file):
    with open(c_file, 'r') as f:
        content = f.read()
    if 'npu_vec_add' not in content:
        target = "extern void matmul_128x128x128_e2e;"
        vec_add_code = """// Matches: declare void @npu_vec_add(ptr addrspace(1), ptr addrspace(1))
void npu_vec_add(uint16_t *matrix_tile, const uint16_t *bias_tile) {
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      float mat_val = f16_to_f32(matrix_tile[i * 16 + j]);
      float bias_val = f16_to_f32(bias_tile[j]);
      matrix_tile[i * 16 + j] = f32_to_f16(mat_val + bias_val);
    }
  }
}

"""
        content = content.replace(target, vec_add_code + target)
        with open(c_file, 'w') as f:
            f.write(content)
        print(f"[SUCCESS] Injected npu_vec_add simulator logic into {c_file}")

# 3. Patch ConvertNPUToLLVM.cpp
cpp_file = 'lib/NPU/ConvertNPUToLLVM.cpp'
if os.path.exists(cpp_file):
    with open(cpp_file, 'r') as f:
        content = f.read()
    if 'VecAddOpLowering' not in content:
        populate_target = "void populateNPUToLLVMConversionPatterns("
        
        struct_code = """
struct VecAddOpLowering : public mlir::ConvertOpToLLVMPattern<mlir::npu::VecAddOp> {
  using ConvertOpToLLVMPattern<mlir::npu::VecAddOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult matchAndRewrite(mlir::npu::VecAddOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto ctx = rewriter.getContext();

    auto llvmVoidTy = mlir::LLVM::LLVMVoidType::get(ctx);
    auto llvmSramPtrTy = mlir::LLVM::LLVMPointerType::get(ctx, /*addressSpace=*/1);
    
    if (!module.lookupSymbol("npu_vec_add")) {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<mlir::LLVM::LLVMFuncOp>(
          op.getLoc(), "npu_vec_add",
          mlir::LLVM::LLVMFunctionType::get(llvmVoidTy, {llvmSramPtrTy, llvmSramPtrTy}));
    }

    mlir::Value matrixPtr = mlir::LLVM::getStridedElementPtr(
        op.getLoc(), adaptor.getMatrixTile().getType(), adaptor.getMatrixTile(),
        rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI64Type(), 0), rewriter);
    
    mlir::Value biasPtr = mlir::LLVM::getStridedElementPtr(
        op.getLoc(), adaptor.getBiasTile().getType(), adaptor.getBiasTile(),
        rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rewriter.getI64Type(), 0), rewriter);

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, llvmVoidTy, mlir::SymbolRefAttr::get(ctx, "npu_vec_add"),
        mlir::ValueRange{matrixPtr, biasPtr});

    return mlir::success();
  }
};
"""
        # Insert struct right before the populate function
        content = content.replace(populate_target, struct_code + "\n" + populate_target)
        
        # Insert patterns.add right inside the populate function block
        add_code = "  patterns.add<VecAddOpLowering>(converter);\n"
        content = re.sub(r'(void populateNPUToLLVMConversionPatterns[^{]*{)', r'\1\n' + add_code, content)
        
        with open(cpp_file, 'w') as f:
            f.write(content)
        print(f"[SUCCESS] Injected VecAddOpLowering into {cpp_file}")

print("Vector engine installation complete!")
