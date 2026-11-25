#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct FuseEpiloguePass : public PassWrapper<FuseEpiloguePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseEpiloguePass)

  StringRef getArgument() const final { return "npu-fuse-epilogue"; }
  StringRef getDescription() const final { return "Manually tile matmul (M,N,K) and bias into 16x16 SRAM blocks."; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, memref::MemRefDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    IRRewriter rewriter(&getContext());

    linalg::MatmulOp matmulOp;
    linalg::GenericOp biasOp;
    func.walk([&](Operation *op) {
      if (auto m = dyn_cast<linalg::MatmulOp>(op)) matmulOp = m;
      if (auto b = dyn_cast<linalg::GenericOp>(op)) {
        if (b.getNumDpsInputs() == 2) biasOp = b;
      }
    });
    if (!matmulOp || !biasOp) return;

    Location loc = matmulOp.getLoc();
    rewriter.setInsertionPoint(matmulOp);

    Value A = matmulOp.getDpsInputOperand(0)->get();
    Value B = matmulOp.getDpsInputOperand(1)->get();
    Value C = matmulOp.getDpsInitOperand(0)->get();
    Value bias = biasOp.getDpsInputOperand(1)->get();

    Value val0 = rewriter.create<arith::ConstantIndexOp>(loc, (int64_t)0);
    Value val128 = rewriter.create<arith::ConstantIndexOp>(loc, (int64_t)128);
    Value val16 = rewriter.create<arith::ConstantIndexOp>(loc, (int64_t)16);

    // 1. M and N Spatial Loops
    auto loopI = rewriter.create<scf::ForOp>(loc, val0, val128, val16, ValueRange{});
    rewriter.setInsertionPointToStart(loopI.getBody());
    
    auto loopJ = rewriter.create<scf::ForOp>(loc, val0, val128, val16, ValueRange{});
    rewriter.setInsertionPointToStart(loopJ.getBody());

    Value i = loopI.getInductionVar();
    Value j = loopJ.getInductionVar();

    auto attr0 = rewriter.getIndexAttr(0);
    auto attr1 = rewriter.getIndexAttr(1);
    auto attr16 = rewriter.getIndexAttr(16);

    auto subC = rewriter.create<memref::SubViewOp>(loc, C, 
        ArrayRef<OpFoldResult>{i, j}, 
        ArrayRef<OpFoldResult>{attr16, attr16}, 
        ArrayRef<OpFoldResult>{attr1, attr1});

    // 2. K Reduction Loop (Accumulating into subC)
    auto loopK = rewriter.create<scf::ForOp>(loc, val0, val128, val16, ValueRange{});
    rewriter.setInsertionPointToStart(loopK.getBody());
    Value k = loopK.getInductionVar();

    auto subA = rewriter.create<memref::SubViewOp>(loc, A, 
        ArrayRef<OpFoldResult>{i, k}, 
        ArrayRef<OpFoldResult>{attr16, attr16}, 
        ArrayRef<OpFoldResult>{attr1, attr1});

    auto subB = rewriter.create<memref::SubViewOp>(loc, B, 
        ArrayRef<OpFoldResult>{k, j}, 
        ArrayRef<OpFoldResult>{attr16, attr16}, 
        ArrayRef<OpFoldResult>{attr1, attr1});

    auto newMatmul = cast<linalg::MatmulOp>(rewriter.clone(*matmulOp.getOperation()));
    newMatmul->setOperand(0, subA);
    newMatmul->setOperand(1, subB);
    newMatmul->setOperand(2, subC);

    // 3. Step OUT of the K Loop (Epilogue happens after accumulation)
    rewriter.setInsertionPointAfter(loopK);

    auto subBias = rewriter.create<memref::SubViewOp>(loc, bias, 
        ArrayRef<OpFoldResult>{j}, 
        ArrayRef<OpFoldResult>{attr16}, 
        ArrayRef<OpFoldResult>{attr1});

    auto newBias = cast<linalg::GenericOp>(rewriter.clone(*biasOp.getOperation()));
    newBias->setOperand(0, subC);
    newBias->setOperand(1, subBias);
    newBias->setOperand(2, subC);

    // 4. Cleanup
    rewriter.eraseOp(biasOp);
    rewriter.eraseOp(matmulOp);
  }
};
} // namespace

namespace mlir {
namespace npu {
// PROPERLY EXPORTED TO LINKER
std::unique_ptr<Pass> createFuseEpiloguePass() {
  return std::make_unique<FuseEpiloguePass>();
}

void registerFuseEpiloguePass() {
  PassRegistration<FuseEpiloguePass>();
}
} // namespace npu
} // namespace mlir
