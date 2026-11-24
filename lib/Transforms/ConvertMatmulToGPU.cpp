#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct MatmulToGPUPattern : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value A = op.getInputs()[0];
    Value B = op.getInputs()[1];
    Value C = op.getOutputs()[0];

    auto aType = dyn_cast<MemRefType>(A.getType());
    if (!aType || !aType.hasStaticShape() || aType.getShape()[0] != 128)
      return failure();

    Type elemType = aType.getElementType();
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value c8 = arith::ConstantIndexOp::create(rewriter, loc, 8);
    Value c16 = arith::ConstantIndexOp::create(rewriter, loc, 16);
    Value c128 = arith::ConstantIndexOp::create(rewriter, loc, 128);
    auto floatTy = cast<FloatType>(elemType);
    Value zero = arith::ConstantFloatOp::create(
        rewriter, loc, floatTy, APFloat::getZero(floatTy.getFloatSemantics()));

    auto wgSpace = gpu::AddressSpaceAttr::get(rewriter.getContext(),
                                             gpu::AddressSpace::Workgroup);
    auto smemType = MemRefType::get({16, 16}, elemType, AffineMap{}, wgSpace);



    auto launchOp = gpu::LaunchOp::create(rewriter, loc, c8, c8, c1, c16, c16, c1);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&launchOp.getBody().front());
    Value smemA = memref::AllocaOp::create(rewriter, loc, smemType);
    Value smemB = memref::AllocaOp::create(rewriter, loc, smemType);

    Value bx = launchOp.getBlockIds().x;
    Value by = launchOp.getBlockIds().y;
    Value tx = launchOp.getThreadIds().x;
    Value ty = launchOp.getThreadIds().y;

    Value globalRow = arith::AddIOp::create(
        rewriter, loc, arith::MulIOp::create(rewriter, loc, by, c16), ty);
    Value globalCol = arith::AddIOp::create(
        rewriter, loc, arith::MulIOp::create(rewriter, loc, bx, c16), tx);

    auto kLoop = scf::ForOp::create(
        rewriter, loc, c0, c128, c16, ValueRange{zero},
        [&](OpBuilder &b, Location l, Value k, ValueRange iterArgs) {
          Value aCol = arith::AddIOp::create(b, l, k, tx);
          Value aVal = memref::LoadOp::create(b, l, A, ValueRange{globalRow, aCol});
          memref::StoreOp::create(b, l, aVal, smemA, ValueRange{ty, tx});

          Value bRow = arith::AddIOp::create(b, l, k, ty);
          Value bVal = memref::LoadOp::create(b, l, B, ValueRange{bRow, globalCol});
          memref::StoreOp::create(b, l, bVal, smemB, ValueRange{ty, tx});

          gpu::BarrierOp::create(b, l);

          auto kkLoop = scf::ForOp::create(
              b, l, c0, c16, c1, iterArgs,
              [&](OpBuilder &ib, Location il, Value kk, ValueRange innerArgs) {
                Value lhs = memref::LoadOp::create(ib, il, smemA, ValueRange{ty, kk});
                Value rhs = memref::LoadOp::create(ib, il, smemB, ValueRange{kk, tx});
                Value prod = arith::MulFOp::create(ib, il, lhs, rhs);
                Value next = arith::AddFOp::create(ib, il, innerArgs[0], prod);
                scf::YieldOp::create(ib, il, next);
              });

          gpu::BarrierOp::create(b, l);
          scf::YieldOp::create(b, l, kkLoop.getResults());
        });

    memref::StoreOp::create(rewriter, loc, kLoop.getResult(0), C,
                            ValueRange{globalRow, globalCol});
    gpu::TerminatorOp::create(rewriter, loc);
    rewriter.setInsertionPointAfter(launchOp);

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertMatmulToGPUPass
    : public PassWrapper<ConvertMatmulToGPUPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertMatmulToGPUPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulToGPUPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createConvertMatmulToGPUPass() {
  return std::make_unique<ConvertMatmulToGPUPass>();
}
} // namespace mlir
