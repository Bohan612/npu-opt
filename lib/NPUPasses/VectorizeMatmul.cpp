#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct VectorizeMatmulPass
    : public PassWrapper<VectorizeMatmulPass, OperationPass<func::FuncOp>> {
public:
  bool lowerContract;
  VectorizeMatmulPass(bool lowerContract = true) : lowerContract(lowerContract) {}
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizeMatmulPass);
  StringRef getArgument() const final { return "npu-vectorize-matmul"; }
  StringRef getDescription() const final {
    return "Tile 64x64 SRAM matmul to 16x16x16 and lower to vector.outerproduct";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);

    SmallVector<linalg::MatmulOp> targetMatmuls;
    funcOp.walk([&](linalg::MatmulOp matmulOp) {
      auto lhsType = dyn_cast<MemRefType>(matmulOp.getInputs()[0].getType());
      if (lhsType && lhsType.getMemorySpaceAsInt() == 1) {
        targetMatmuls.push_back(matmulOp);
      }
    });

    SmallVector<OpFoldResult> tileSizes = {
        rewriter.getIndexAttr(16),
        rewriter.getIndexAttr(16),
        rewriter.getIndexAttr(16),
    };
    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes);

    for (linalg::MatmulOp matmulOp : targetMatmuls) {
      rewriter.setInsertionPoint(matmulOp);
      FailureOr<scf::SCFTilingResult> tiledResult = scf::tileUsingSCF(
          rewriter, cast<TilingInterface>(matmulOp.getOperation()),
          tilingOptions);
      if (failed(tiledResult)) {
        signalPassFailure();
        return;
      }

      rewriter.replaceOp(matmulOp, tiledResult->replacements);

      for (Operation *tiledOp : tiledResult->tiledOps) {
        if (auto innerMatmul = dyn_cast<linalg::MatmulOp>(tiledOp)) {
          if (failed(linalg::vectorize(rewriter, innerMatmul))) {
            signalPassFailure();
            return;
          }
          rewriter.eraseOp(innerMatmul);
          if (false) {
            signalPassFailure();
            return;
          }
        }
      }
    }

    RewritePatternSet patterns(ctx);
    
    if (lowerContract)
      vector::populateVectorContractLoweringPatterns(
          patterns, vector::VectorContractLowering::OuterProduct);
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    vector::populateVectorReductionToContractPatterns(patterns);
    vector::populateSinkVectorOpsPatterns(patterns);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace mlir::npu {
std::unique_ptr<Pass> createVectorizeMatmulPass(bool lowerContract = true) {
  return std::make_unique<VectorizeMatmulPass>(lowerContract);
}
} // namespace mlir::npu
