//===- TileMatmul.cpp - K-Dimension Tiling & SRAM Scratchpad Promotion ----===//
//
// Tiles linalg.matmul along the reduction dimension (K) using TilingInterface
// and promotes tiled DDR subviews into scratchpad SRAM (memory_space = 1) so
// downstream passes can apply ping-pong async DMA pipelining.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TileMatmulPass
    : public PassWrapper<TileMatmulPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileMatmulPass)

  StringRef getArgument() const final { return "npu-tile-matmul"; }
  StringRef getDescription() const final {
    return "Tile linalg.matmul along K and promote operands to SRAM "
           "(memory_space = 1)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    SmallVector<linalg::MatmulOp> matmuls;
    funcOp.walk([&](linalg::MatmulOp op) {
      // Only tile matmuls reading directly from global DDR (memory_space == 0).
      auto lhsType = dyn_cast<MemRefType>(op.getInputs()[0].getType());
      auto rhsType = dyn_cast<MemRefType>(op.getInputs()[1].getType());
      if (lhsType && rhsType && lhsType.hasStaticShape() &&
          rhsType.hasStaticShape() && lhsType.getMemorySpaceAsInt() == 0 &&
          rhsType.getMemorySpaceAsInt() == 0) {
        matmuls.push_back(op);
      }
    });

    for (linalg::MatmulOp matmulOp : matmuls) {
      if (failed(tileAndPromoteToSRAM(rewriter, matmulOp))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult tileAndPromoteToSRAM(IRRewriter &rewriter,
                                     linalg::MatmulOp matmulOp) {
    auto tilingOp = cast<TilingInterface>(matmulOp.getOperation());

    // Tile only the reduction dimension K = 64 (M = 0, N = 0, K = 64).
    SmallVector<OpFoldResult> tileSizes = {
        rewriter.getIndexAttr(0),
        rewriter.getIndexAttr(0),
        rewriter.getIndexAttr(64),
    };

    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes);

    rewriter.setInsertionPoint(matmulOp);
    FailureOr<scf::SCFTilingResult> tiledResult =
        scf::tileUsingSCF(rewriter, tilingOp, tilingOptions);
    if (failed(tiledResult))
      return failure();

    // Replace the untiled matmul with the newly generated tiled scf.for loop.
    rewriter.replaceOp(matmulOp, tiledResult->replacements);

    if (tiledResult->loops.empty() || tiledResult->tiledOps.empty())
      return failure();

    auto forOp = cast<scf::ForOp>(tiledResult->loops.front());
    auto tiledMatmul = cast<linalg::MatmulOp>(tiledResult->tiledOps.front());

    // Promote the two tiled DDR input subviews into scratchpad SRAM (space 1).
    Location loc = tiledMatmul.getLoc();
    for (unsigned idx = 0; idx < 2; ++idx) {
      Value ddrSlice = tiledMatmul.getInputs()[idx];
      auto sliceType = cast<MemRefType>(ddrSlice.getType());

      // Allocate static contiguous SRAM buffer outside the loop.
      rewriter.setInsertionPoint(forOp);
      auto sramType = MemRefType::get(sliceType.getShape(),
                                      sliceType.getElementType(),
                                      MemRefLayoutAttrInterface{},
                                      rewriter.getI32IntegerAttr(1));
      Value sramAlloc = memref::AllocOp::create(rewriter, loc, sramType);

      // Copy DDR slice -> SRAM buffer right before the tiled matmul.
      rewriter.setInsertionPoint(tiledMatmul);
      memref::CopyOp::create(rewriter, loc, ddrSlice, sramAlloc);

      // Point the tiled matmul input to the fast SRAM buffer.
      tiledMatmul.setOperand(idx, sramAlloc);
    }

    return success();
  }
};

} // namespace

namespace mlir::npu {
std::unique_ptr<Pass> createTileMatmulPass() {
  return std::make_unique<TileMatmulPass>();
}
} // namespace mlir::npu
