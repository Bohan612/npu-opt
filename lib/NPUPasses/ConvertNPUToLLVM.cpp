#include "NPU/NPUDialect.h"
#include "NPU/NPUOps.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static LLVM::LLVMFuncOp getOrInsertRuntimeFn(ModuleOp module,
                                             OpBuilder &builder,
                                             StringRef name,
                                             Type resultType,
                                             ArrayRef<Type> argTypes) {
  if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return fn;
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto fnType = LLVM::LLVMFunctionType::get(resultType, argTypes);
  return LLVM::LLVMFuncOp::create(builder, module.getLoc(), name, fnType);
}

struct DMALoadToLLVMPattern : public ConvertOpToLLVMPattern<npu::DMALoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(npu::DMALoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
        auto i64Type = rewriter.getI64Type();

    MemRefDescriptor srcDesc(adaptor.getSrc());
    MemRefDescriptor dstDesc(adaptor.getDst());
    Value srcPtr = srcDesc.alignedPtr(rewriter, loc);
    Value dstPtr = dstDesc.alignedPtr(rewriter, loc);

    auto memrefType = cast<MemRefType>(op.getSrc().getType());
    int64_t numBytes = memrefType.getNumElements() *
                       (memrefType.getElementTypeBitWidth() / 8);
    Value bytesVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(numBytes));

    auto fn = getOrInsertRuntimeFn(module, rewriter, "npu_dma_push", i64Type,
                                   {srcPtr.getType(), dstPtr.getType(), i64Type});
    auto call = LLVM::CallOp::create(
        rewriter, loc, fn, ValueRange{srcPtr, dstPtr, bytesVal});

    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

struct DMAWaitToLLVMPattern : public ConvertOpToLLVMPattern<npu::DMAWaitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(npu::DMAWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto i64Type = rewriter.getI64Type();

    auto fn = getOrInsertRuntimeFn(module, rewriter, "npu_dma_wait", voidType,
                                   {i64Type});
    LLVM::CallOp::create(rewriter, loc, fn, ValueRange{adaptor.getToken()});
    rewriter.eraseOp(op);
    return success();
  }
};

struct MMAToLLVMPattern : public ConvertOpToLLVMPattern<npu::MMAOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(npu::MMAOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    Type vecType = typeConverter->convertType(op.getResult().getType());
    if (!vecType)
      return failure();

    auto fn = getOrInsertRuntimeFn(module, rewriter, "npu_cube_mma", vecType,
                                   {vecType, vecType, vecType});
    auto call = LLVM::CallOp::create(
        rewriter, loc, fn,
        ValueRange{adaptor.getLhs(), adaptor.getRhs(), adaptor.getAcc()});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

struct ConvertNPUToLLVMPass
    : public PassWrapper<ConvertNPUToLLVMPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertNPUToLLVMPass);

  StringRef getArgument() const final { return "convert-npu-to-llvm"; }
  StringRef getDescription() const final {
    return "Lower npu dialect ops and !npu.token to LLVM dialect runtime calls";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, npu::NPUDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    LLVMTypeConverter typeConverter(ctx);
    typeConverter.addConversion([ctx](npu::TokenType) -> Type {
      return IntegerType::get(ctx, 64);
    });

    LLVMConversionTarget target(*ctx);
    target.addLegalOp<ModuleOp>();
    target.addIllegalDialect<npu::NPUDialect>();

    RewritePatternSet patterns(ctx);
    patterns.add<DMALoadToLLVMPattern, DMAWaitToLLVMPattern, MMAToLLVMPattern>(
        typeConverter);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::npu {
std::unique_ptr<Pass> createConvertNPUToLLVMPass() {
  return std::make_unique<ConvertNPUToLLVMPass>();
}
} // namespace mlir::npu
