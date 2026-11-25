#include "NPU/NPUDialect.h"
#include "NPU/NPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct DMAStartToNPULoadPattern : public OpConversionPattern<memref::DmaStartOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DmaStartOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value tag = op.getTagMemRef();
    auto tokenType = npu::TokenType::get(rewriter.getContext());
    auto loadOp = npu::DMALoadOp::create(
        rewriter, op.getLoc(), tokenType, adaptor.getOperands()[0],
        adaptor.getOperands()[1 + op.getSrcMemRefRank()]);

    // Find the matching dma_wait for this dma_start (before any next dma_start on the same tag)
    if (auto funcOp = op->getParentOfType<func::FuncOp>()) {
      funcOp.walk([&](memref::DmaWaitOp waitOp) {
        if (waitOp.getTagMemRef() == tag &&
            llvm::equal(waitOp.getTagIndices(), op.getTagIndices())) {
          { OpBuilder::InsertionGuard g(rewriter); rewriter.setInsertionPoint(waitOp); rewriter.replaceOpWithNewOp<npu::DMAWaitOp>(waitOp, loadOp.getToken()); }
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct DMAWaitToNPUWaitPattern : public OpConversionPattern<memref::DmaWaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DmaWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value token = adaptor.getTagMemRef();
    if (!isa<npu::TokenType>(token.getType()))
      return failure();
    rewriter.replaceOpWithNewOp<npu::DMAWaitOp>(op, token);
    return success();
  }
};

struct VectorContractToNPUMMAPattern
    : public OpConversionPattern<vector::ContractionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ContractionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto accType = dyn_cast<VectorType>(adaptor.getAcc().getType());
    if (!accType || !accType.getElementType().isF16())
      return failure();
    rewriter.replaceOpWithNewOp<npu::MMAOp>(
        op, accType, adaptor.getLhs(), adaptor.getRhs(), adaptor.getAcc());
    return success();
  }
};


struct GenericAddToNPUVecAddPattern : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Ensure it is our 2-input bias addition
    if (op.getNumDpsInputs() != 2) return failure();
    
    // Using getOperands() avoids MLIR version API mismatches.
    // Operand 0 is the Matrix C tile, Operand 1 is the Bias tile.
    rewriter.replaceOpWithNewOp<npu::VecAddOp>(op, adaptor.getOperands()[0], adaptor.getOperands()[1]);
    return success();
  }
};

struct ConvertToNPUDialectPass
    : public PassWrapper<ConvertToNPUDialectPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertToNPUDialectPass);
  StringRef getArgument() const final { return "convert-to-npu"; }
  StringRef getDescription() const final {
    return "Lower memref DMA ops and vector.contract into custom npu dialect ops";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<npu::NPUDialect, memref::MemRefDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([ctx](MemRefType t) -> std::optional<Type> {
      if (t.getRank() == 1 && t.getNumElements() == 1)
        return npu::TokenType::get(ctx);
      return t;
    });

    ConversionTarget target(*ctx);
    target.addLegalDialect<npu::NPUDialect>();
    target.addIllegalOp<memref::DmaStartOp, memref::DmaWaitOp, vector::ContractionOp>();
    target.addLegalOp<func::FuncOp, func::ReturnOp>();
    

    RewritePatternSet patterns(ctx);
    patterns.add<DMAStartToNPULoadPattern, DMAWaitToNPUWaitPattern,
                 VectorContractToNPUMMAPattern, GenericAddToNPUVecAddPattern>(typeConverter, ctx);

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    funcOp.walk([](memref::AllocOp alloc) {
      MemRefType t = alloc.getType();
      if (isa<IntegerType, IndexType>(t.getElementType()) && alloc.use_empty())
        alloc.erase();
    });
  }
};
} // namespace

namespace mlir::npu {
std::unique_ptr<Pass> createConvertToNPUDialectPass() {
  return std::make_unique<ConvertToNPUDialectPass>();
}
} // namespace mlir::npu
