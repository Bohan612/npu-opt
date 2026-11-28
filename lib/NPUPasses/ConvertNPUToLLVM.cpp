#include "NPU/NPUDialect.h"
#include "NPU/NPUOps.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

static Value toDefaultAddrSpacePtr(ConversionPatternRewriter &rewriter,
                                   Location loc, Value ptr) {
  auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(ptr.getType());
  if (!ptrTy || ptrTy.getAddressSpace() == 0)
    return ptr;
  auto flatPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
  return LLVM::AddrSpaceCastOp::create(rewriter, loc, flatPtrTy, ptr);
}

struct DMALoadToLLVMPattern : public ConvertOpToLLVMPattern<npu::DMALoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(npu::DMALoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Type = rewriter.getI64Type();
    auto flatPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);

    MemRefDescriptor srcDesc(adaptor.getSrc());
    MemRefDescriptor dstDesc(adaptor.getDst());
    auto memrefType = cast<MemRefType>(op.getSrc().getType());
    auto dstMemRefType = cast<MemRefType>(op.getDst().getType());
    Type srcElemTy = typeConverter->convertType(memrefType.getElementType());
    Type dstElemTy = typeConverter->convertType(dstMemRefType.getElementType());

    Value srcPtr = toDefaultAddrSpacePtr(
        rewriter, loc,
        LLVM::GEPOp::create(rewriter, loc, srcDesc.alignedPtr(rewriter, loc).getType(),
                            srcElemTy, srcDesc.alignedPtr(rewriter, loc),
                            ValueRange{srcDesc.offset(rewriter, loc)}));
    Value dstPtr = toDefaultAddrSpacePtr(
        rewriter, loc,
        LLVM::GEPOp::create(rewriter, loc, dstDesc.alignedPtr(rewriter, loc).getType(),
                            dstElemTy, dstDesc.alignedPtr(rewriter, loc),
                            ValueRange{dstDesc.offset(rewriter, loc)}));

    int64_t numBytes = memrefType.getNumElements() *
                       (memrefType.getElementTypeBitWidth() / 8);
    Value bytesVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(numBytes));

    auto fn = getOrInsertRuntimeFn(module, rewriter, "npu_dma_push", i64Type,
                                   {flatPtrTy, flatPtrTy, i64Type});
    auto call = LLVM::CallOp::create(
        rewriter, loc, fn, ValueRange{srcPtr, dstPtr, bytesVal});

    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

struct DMALoad2DToLLVMPattern
    : public ConvertOpToLLVMPattern<npu::DMALoad2DOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(npu::DMALoad2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Type = rewriter.getI64Type();
    auto flatPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);

    MemRefDescriptor srcDesc(adaptor.getSrc());
    MemRefDescriptor dstDesc(adaptor.getDst());
    auto srcMemRefType = cast<MemRefType>(op.getSrc().getType());
    auto dstMemRefType = cast<MemRefType>(op.getDst().getType());
    Type srcElemTy = typeConverter->convertType(srcMemRefType.getElementType());
    Type dstElemTy = typeConverter->convertType(dstMemRefType.getElementType());

    Value srcPtr = toDefaultAddrSpacePtr(
        rewriter, loc,
        LLVM::GEPOp::create(rewriter, loc, srcDesc.alignedPtr(rewriter, loc).getType(),
                            srcElemTy, srcDesc.alignedPtr(rewriter, loc),
                            ValueRange{srcDesc.offset(rewriter, loc)}));
    Value dstPtr = toDefaultAddrSpacePtr(
        rewriter, loc,
        LLVM::GEPOp::create(rewriter, loc, dstDesc.alignedPtr(rewriter, loc).getType(),
                            dstElemTy, dstDesc.alignedPtr(rewriter, loc),
                            ValueRange{dstDesc.offset(rewriter, loc)}));

    auto fn = getOrInsertRuntimeFn(
        module, rewriter, "npu_dma_load_2d", i64Type,
        {flatPtrTy, flatPtrTy, i64Type, i64Type, i64Type});
    auto call = LLVM::CallOp::create(
        rewriter, loc, fn,
        ValueRange{srcPtr, dstPtr, adaptor.getRows(), adaptor.getCols(),
                   adaptor.getStride()});

    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

struct DMAStore2DToLLVMPattern
    : public ConvertOpToLLVMPattern<npu::DMAStore2DOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(npu::DMAStore2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto i64Type = rewriter.getI64Type();
    auto flatPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);

    MemRefDescriptor srcDesc(adaptor.getSrc());
    MemRefDescriptor dstDesc(adaptor.getDst());
    auto srcMemRefType = cast<MemRefType>(op.getSrc().getType());
    auto dstMemRefType = cast<MemRefType>(op.getDst().getType());
    Type srcElemTy = typeConverter->convertType(srcMemRefType.getElementType());
    Type dstElemTy = typeConverter->convertType(dstMemRefType.getElementType());

    Value srcPtr = toDefaultAddrSpacePtr(
        rewriter, loc,
        LLVM::GEPOp::create(rewriter, loc, srcDesc.alignedPtr(rewriter, loc).getType(),
                            srcElemTy, srcDesc.alignedPtr(rewriter, loc),
                            ValueRange{srcDesc.offset(rewriter, loc)}));
    Value dstPtr = toDefaultAddrSpacePtr(
        rewriter, loc,
        LLVM::GEPOp::create(rewriter, loc, dstDesc.alignedPtr(rewriter, loc).getType(),
                            dstElemTy, dstDesc.alignedPtr(rewriter, loc),
                            ValueRange{dstDesc.offset(rewriter, loc)}));

    auto fn = getOrInsertRuntimeFn(
        module, rewriter, "npu_dma_store_2d", i64Type,
        {flatPtrTy, flatPtrTy, i64Type, i64Type, i64Type});
    auto call = LLVM::CallOp::create(
        rewriter, loc, fn,
        ValueRange{srcPtr, dstPtr, adaptor.getRows(), adaptor.getCols(),
                   adaptor.getStride()});

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

    auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto flatPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
    auto one = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                        rewriter.getI64IntegerAttr(1));

    auto spillToFlatPtr = [&](Value val) -> Value {
      auto alloca = LLVM::AllocaOp::create(rewriter, loc, flatPtrTy, vecType, one);
      LLVM::StoreOp::create(rewriter, loc, val, alloca);
      return toDefaultAddrSpacePtr(rewriter, loc, alloca);
    };

    Value aPtr = spillToFlatPtr(adaptor.getOperands()[0]);
    Value bPtr = spillToFlatPtr(adaptor.getOperands()[1]);
    Value cPtr = spillToFlatPtr(adaptor.getAcc());
    auto outAlloca = LLVM::AllocaOp::create(rewriter, loc, flatPtrTy, vecType, one);
    Value outPtr = toDefaultAddrSpacePtr(rewriter, loc, outAlloca);

    auto fn = getOrInsertRuntimeFn(
        module, rewriter, "npu_cube_mma", voidType,
        {flatPtrTy, flatPtrTy, flatPtrTy, flatPtrTy});
    LLVM::CallOp::create(rewriter, loc, fn, ValueRange{aPtr, bPtr, cPtr, outPtr});

    Value outVal = LLVM::LoadOp::create(rewriter, loc, vecType, outAlloca);
    rewriter.replaceOp(op, outVal);
    return success();
  }
};

struct NPUVecAddToLLVMPattern : public ConvertOpToLLVMPattern<npu::VecAddOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(npu::VecAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());

    MemRefDescriptor lhsDesc(adaptor.getOperands()[0]);
    MemRefDescriptor rhsDesc(adaptor.getOperands()[1]);

    auto lhsMemRefType = cast<MemRefType>(op->getOperand(0).getType());
    auto rhsMemRefType = cast<MemRefType>(op->getOperand(1).getType());

    Type lhsElemTy = typeConverter->convertType(lhsMemRefType.getElementType());
    Type rhsElemTy = typeConverter->convertType(rhsMemRefType.getElementType());

    Value lhsPtr = LLVM::GEPOp::create(
        rewriter, loc, lhsDesc.alignedPtr(rewriter, loc).getType(), lhsElemTy,
        lhsDesc.alignedPtr(rewriter, loc), ValueRange{lhsDesc.offset(rewriter, loc)});
    Value rhsPtr = LLVM::GEPOp::create(
        rewriter, loc, rhsDesc.alignedPtr(rewriter, loc).getType(), rhsElemTy,
        rhsDesc.alignedPtr(rewriter, loc), ValueRange{rhsDesc.offset(rewriter, loc)});

    auto fn = getOrInsertRuntimeFn(module, rewriter, "npu_vec_add_c_harness", voidType,
                                   {lhsPtr.getType(), rhsPtr.getType()});

    LLVM::CallOp::create(rewriter, loc, fn, ValueRange{lhsPtr, rhsPtr});

    rewriter.eraseOp(op);
    return success();
  }
};

struct LinalgMatmulToLLVMPattern : public ConvertOpToLLVMPattern<linalg::MatmulOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());

    MemRefDescriptor aDesc(adaptor.getOperands()[0]);
    MemRefDescriptor bDesc(adaptor.getOperands()[1]);
    MemRefDescriptor cDesc(adaptor.getOperands()[2]);

    auto aMemRefType = cast<MemRefType>(op->getOperand(0).getType());
    auto bMemRefType = cast<MemRefType>(op->getOperand(1).getType());
    auto cMemRefType = cast<MemRefType>(op->getOperand(2).getType());

    Type aElemTy = typeConverter->convertType(aMemRefType.getElementType());
    Type bElemTy = typeConverter->convertType(bMemRefType.getElementType());
    Type cElemTy = typeConverter->convertType(cMemRefType.getElementType());

    Value aPtr = LLVM::GEPOp::create(
        rewriter, loc, aDesc.alignedPtr(rewriter, loc).getType(), aElemTy,
        aDesc.alignedPtr(rewriter, loc), ValueRange{aDesc.offset(rewriter, loc)});
    Value bPtr = LLVM::GEPOp::create(
        rewriter, loc, bDesc.alignedPtr(rewriter, loc).getType(), bElemTy,
        bDesc.alignedPtr(rewriter, loc), ValueRange{bDesc.offset(rewriter, loc)});
    Value cPtr = LLVM::GEPOp::create(
        rewriter, loc, cDesc.alignedPtr(rewriter, loc).getType(), cElemTy,
        cDesc.alignedPtr(rewriter, loc), ValueRange{cDesc.offset(rewriter, loc)});

    auto fn = getOrInsertRuntimeFn(module, rewriter, "npu_cube_mma_c_harness", voidType,
                                   {aPtr.getType(), bPtr.getType(), cPtr.getType()});
    LLVM::CallOp::create(rewriter, loc, fn, ValueRange{aPtr, bPtr, cPtr});

    rewriter.eraseOp(op);
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
    patterns.add<DMALoadToLLVMPattern, DMALoad2DToLLVMPattern,
                 DMAStore2DToLLVMPattern, DMAWaitToLLVMPattern,
                 MMAToLLVMPattern, NPUVecAddToLLVMPattern,
                 LinalgMatmulToLLVMPattern>(typeConverter);

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

namespace mlir {
namespace npu {
void registerConvertNPUToLLVMPass() { PassRegistration<ConvertNPUToLLVMPass>(); }
} // namespace npu
} // namespace mlir
