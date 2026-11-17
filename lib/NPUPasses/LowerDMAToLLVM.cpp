//===- LowerDMAToLLVM.cpp - Lower memref.dma_start/wait to LLVM Calls -----===//
//
// Converts asynchronous memref DMA operations into external NPU runtime LLVM
// function calls (@npu_dma_start / @npu_dma_wait) using LLVMTypeConverter.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static LLVM::LLVMFuncOp getOrInsertDMAFunc(OpBuilder &b, ModuleOp module,
                                           StringRef name,
                                           LLVM::LLVMFunctionType fnType) {
  if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return existing;
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(module.getBody());
  return LLVM::LLVMFuncOp::create(b, module.getLoc(), name, fnType);
}

struct DmaStartOpLowering : public ConvertOpToLLVMPattern<memref::DmaStartOp> {
  using ConvertOpToLLVMPattern<memref::DmaStartOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::DmaStartOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Type = rewriter.getI64Type();
    auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());

    // DmaStartOp operand layout:
    // [0] srcMemRef, [1..srcRank] srcIndices,
    // [1+srcRank] dstMemRef, [2+srcRank..1+srcRank+dstRank] dstIndices,
    // [2+srcRank+dstRank] numElements,
    // [3+srcRank+dstRank] tagMemRef, [4+srcRank+dstRank..] tagIndices
    ValueRange operands = adaptor.getOperands();
    unsigned srcRank = op.getSrcMemRefRank();
    unsigned dstRank = op.getDstMemRefRank();

    unsigned dstIdx = 1 + srcRank;
    unsigned numElemsIdx = 2 + srcRank + dstRank;
    unsigned tagIdx = 3 + srcRank + dstRank;

    MemRefDescriptor srcDesc(operands[0]);
    MemRefDescriptor dstDesc(operands[dstIdx]);
    MemRefDescriptor tagDesc(operands[tagIdx]);

    Value srcPtr = srcDesc.alignedPtr(rewriter, loc);
    Value dstPtr = dstDesc.alignedPtr(rewriter, loc);
    Value numElems = operands[numElemsIdx];
    Value tagPtr = tagDesc.alignedPtr(rewriter, loc);

    // void @npu_dma_start(ptr src, ptr<1> dst, i64 num_elements, ptr<1> tag)
    auto fnType = LLVM::LLVMFunctionType::get(
        voidType,
        {srcPtr.getType(), dstPtr.getType(), i64Type, tagPtr.getType()},
        /*isVarArg=*/false);
    LLVM::LLVMFuncOp fn =
        getOrInsertDMAFunc(rewriter, module, "npu_dma_start", fnType);

    LLVM::CallOp::create(rewriter, loc, fn,
                         ValueRange{srcPtr, dstPtr, numElems, tagPtr});
    rewriter.eraseOp(op);
    return success();
  }
};

struct DmaWaitOpLowering : public ConvertOpToLLVMPattern<memref::DmaWaitOp> {
  using ConvertOpToLLVMPattern<memref::DmaWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::DmaWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Type = rewriter.getI64Type();
    auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());

    // DmaWaitOp operand layout:
    // [0] tagMemRef, [1..tagRank] tagIndices, [1+tagRank] numElements
    ValueRange operands = adaptor.getOperands();
    unsigned tagRank = op.getTagMemRefRank();

    MemRefDescriptor tagDesc(operands[0]);
    Value tagPtr = tagDesc.alignedPtr(rewriter, loc);
    Value numElems = operands[1 + tagRank];

    // void @npu_dma_wait(ptr<1> tag, i64 num_elements)
    auto fnType = LLVM::LLVMFunctionType::get(
        voidType, {tagPtr.getType(), i64Type}, /*isVarArg=*/false);
    LLVM::LLVMFuncOp fn =
        getOrInsertDMAFunc(rewriter, module, "npu_dma_wait", fnType);

    LLVM::CallOp::create(rewriter, loc, fn, ValueRange{tagPtr, numElems});
    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerDMAToLLVMPass
    : public PassWrapper<LowerDMAToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerDMAToLLVMPass)

  StringRef getArgument() const final { return "npu-lower-dma-to-llvm"; }
  StringRef getDescription() const final {
    return "Lower memref.dma_start and memref.dma_wait to NPU runtime LLVM calls";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    LLVMTypeConverter typeConverter(&getContext());

    RewritePatternSet patterns(&getContext());
    patterns.add<DmaStartOpLowering, DmaWaitOpLowering>(typeConverter);

    LLVMConversionTarget target(getContext());
    target.addIllegalOp<memref::DmaStartOp, memref::DmaWaitOp>();
    target.addLegalDialect<LLVM::LLVMDialect>();

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::npu {
std::unique_ptr<Pass> createLowerDMAToLLVMPass() {
  return std::make_unique<LowerDMAToLLVMPass>();
}
} // namespace mlir::npu
