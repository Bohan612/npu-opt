//===- TileMatmul.cpp - Macro-Tiling & SRAM Scratchpad Promotion ---------===//
//
// Tiles linalg.matmul across (M, N, K) macro-tiles using TilingInterface
// and promotes tiled DDR subviews into scratchpad SRAM (memory_space = 1) so
// downstream passes can apply ping-pong async DMA pipelining and 16x16 MMA.
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

  TileMatmulPass() = default;
  TileMatmulPass(const TileMatmulPass &other) {
    macroM = other.macroM.getValue();
    macroN = other.macroN.getValue();
    macroK = other.macroK.getValue();
  }

  Option<int64_t> macroM{*this, "macro-m",
                         llvm::cl::desc("Macro-tile size along M (0 = untiled)"),
                         llvm::cl::init(64)};
  Option<int64_t> macroN{*this, "macro-n",
                         llvm::cl::desc("Macro-tile size along N (0 = untiled)"),
                         llvm::cl::init(64)};
  Option<int64_t> macroK{*this, "macro-k",
                         llvm::cl::desc("Macro-tile size along K (0 = untiled)"),
                         llvm::cl::init(64)};

  StringRef getArgument() const final { return "npu-tile-matmul"; }
  StringRef getDescription() const final {
    return "Tile linalg.matmul into SRAM macro-tiles (memory_space = 1)";
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

    SmallVector<OpFoldResult> tileSizes = {
        rewriter.getIndexAttr(macroM.getValue()),
        rewriter.getIndexAttr(macroN.getValue()),
        rewriter.getIndexAttr(macroK.getValue()),
    };

    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes);

    rewriter.setInsertionPoint(matmulOp);
    FailureOr<scf::SCFTilingResult> tiledResult =
        scf::tileUsingSCF(rewriter, tilingOp, tilingOptions);
    if (failed(tiledResult))
      return failure();

    rewriter.replaceOp(matmulOp, tiledResult->replacements);

    if (tiledResult->loops.empty() || tiledResult->tiledOps.empty())
      return failure();

    Operation *outerLoop = tiledResult->loops.front();
    auto tiledMatmul = cast<linalg::MatmulOp>(tiledResult->tiledOps.front());

    Location loc = tiledMatmul.getLoc();
    for (unsigned idx = 0; idx < 2; ++idx) {
      Value ddrSlice = tiledMatmul.getInputs()[idx];
      auto sliceType = cast<MemRefType>(ddrSlice.getType());

      rewriter.setInsertionPoint(outerLoop);
      auto sramType = MemRefType::get(sliceType.getShape(),
                                      sliceType.getElementType(),
                                      MemRefLayoutAttrInterface{},
                                      rewriter.getI32IntegerAttr(1));
      Value sramAlloc = memref::AllocOp::create(rewriter, loc, sramType);

      rewriter.setInsertionPoint(tiledMatmul);
      memref::CopyOp::create(rewriter, loc, ddrSlice, sramAlloc);

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
