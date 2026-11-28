#include "NPU/NPUDialect.h"
#include "NPU/NPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static Value getPingPongBankView(IRRewriter &rewriter, Location loc,
                                 Value pingPongBuf, Value bankIdx,
                                 OpFoldResult attr0, OpFoldResult attr16,
                                 OpFoldResult attr1) {
  auto srcTy = cast<MemRefType>(pingPongBuf.getType());
  SmallVector<OpFoldResult, 3> offsets = {OpFoldResult(bankIdx), attr0, attr0};
  SmallVector<OpFoldResult, 3> sizes = {attr1, attr16, attr16};
  SmallVector<OpFoldResult, 3> strides = {attr1, attr1, attr1};
  auto dstTy = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
      {16, 16}, srcTy, offsets, sizes, strides));
  return memref::SubViewOp::create(rewriter, loc, dstTy, pingPongBuf, offsets,
                                   sizes, strides);
}

struct FuseEpiloguePass
    : public PassWrapper<FuseEpiloguePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseEpiloguePass)

  StringRef getArgument() const final { return "npu-fuse-epilogue"; }
  StringRef getDescription() const final {
    return "Tile matmul (M,N,K) and bias into 16x16 SRAM blocks with ping-pong 2D DMA.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, memref::MemRefDialect,
                    arith::ArithDialect, npu::NPUDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    IRRewriter rewriter(&getContext());

    linalg::MatmulOp matmulOp;
    linalg::GenericOp biasOp;
    func.walk([&](Operation *op) {
      if (auto m = dyn_cast<linalg::MatmulOp>(op))
        matmulOp = m;
      if (auto b = dyn_cast<linalg::GenericOp>(op)) {
        if (b.getNumDpsInputs() == 2)
          biasOp = b;
      }
    });
    if (!matmulOp || !biasOp)
      return;

    linalg::GenericOp reluOp;
    func.walk([&](linalg::GenericOp gen) {
      if (gen == biasOp || gen.getNumDpsInputs() != 1)
        return;
      bool hasMax = false;
      gen.getBody()->walk([&](Operation *inner) {
        if (isa<arith::MaximumFOp, arith::MaxNumFOp>(inner))
          hasMax = true;
      });
      if (hasMax)
        reluOp = gen;
    });

    Location loc = matmulOp.getLoc();
    rewriter.setInsertionPoint(matmulOp);

    Value A = matmulOp.getDpsInputOperand(0)->get();
    Value B = matmulOp.getDpsInputOperand(1)->get();
    Value C = matmulOp.getDpsInitOperand(0)->get();
    Value bias = biasOp.getDpsInputOperand(1)->get();

    auto f16Ty = rewriter.getF16Type();
    auto tokenTy = npu::TokenType::get(&getContext());
    auto sramSpace = rewriter.getI64IntegerAttr(1);

    auto pingPongTileType =
        MemRefType::get({2, 16, 16}, f16Ty, AffineMap{}, sramSpace);
    auto tileType = MemRefType::get({16, 16}, f16Ty, AffineMap{}, sramSpace);
    auto biasSramType = MemRefType::get({16}, f16Ty, AffineMap{}, sramSpace);

    Value pingPongA = memref::AllocOp::create(rewriter, loc, pingPongTileType);
    Value pingPongB = memref::AllocOp::create(rewriter, loc, pingPongTileType);
    Value sramC = memref::AllocOp::create(rewriter, loc, tileType);
    Value sramBias = memref::AllocOp::create(rewriter, loc, biasSramType);

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value c16 = arith::ConstantIndexOp::create(rewriter, loc, 16);
    Value c128 = arith::ConstantIndexOp::create(rewriter, loc, 128);

    Value zeroF16 = arith::ConstantFloatOp::create(
        rewriter, loc, cast<FloatType>(f16Ty),
        APFloat::getZero(cast<FloatType>(f16Ty).getFloatSemantics()));

    auto attr0 = rewriter.getIndexAttr(0);
    auto attr1 = rewriter.getIndexAttr(1);
    auto attr16 = rewriter.getIndexAttr(16);

    auto loopI = scf::ForOp::create(rewriter, loc, c0, c128, c16, ValueRange{});
    rewriter.setInsertionPointToStart(loopI.getBody());

    auto loopJ = scf::ForOp::create(rewriter, loc, c0, c128, c16, ValueRange{});
    rewriter.setInsertionPointToStart(loopJ.getBody());

    Value i = loopI.getInductionVar();
    Value j = loopJ.getInductionVar();

    auto subBias = memref::SubViewOp::create(
        rewriter, loc, bias, ArrayRef<OpFoldResult>{j},
        ArrayRef<OpFoldResult>{attr16}, ArrayRef<OpFoldResult>{attr1});
    Value tokBias = npu::DMALoad2DOp::create(
                        rewriter, loc, tokenTy, subBias, sramBias, c1, c16, c1)
                        .getToken();

    auto subC = memref::SubViewOp::create(
        rewriter, loc, C, ArrayRef<OpFoldResult>{i, j},
        ArrayRef<OpFoldResult>{attr16, attr16},
        ArrayRef<OpFoldResult>{attr1, attr1});

    // Zero sramC tile for this (i, j) output block.
    auto zeroRow = scf::ForOp::create(rewriter, loc, c0, c16, c1, ValueRange{});
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(zeroRow.getBody());
      Value r = zeroRow.getInductionVar();
      auto zeroCol =
          scf::ForOp::create(rewriter, loc, c0, c16, c1, ValueRange{});
      rewriter.setInsertionPointToStart(zeroCol.getBody());
      Value col = zeroCol.getInductionVar();
      memref::StoreOp::create(rewriter, loc, zeroF16, sramC, ValueRange{r, col});
    }

    // Prologue: load k = 0 tile into bank 0 and wait for completion.
    Value bank0A = getPingPongBankView(rewriter, loc, pingPongA, c0, attr0,
                                       attr16, attr1);
    Value bank0B = getPingPongBankView(rewriter, loc, pingPongB, c0, attr0,
                                       attr16, attr1);
    auto subA0 = memref::SubViewOp::create(
        rewriter, loc, A, ArrayRef<OpFoldResult>{i, attr0},
        ArrayRef<OpFoldResult>{attr16, attr16},
        ArrayRef<OpFoldResult>{attr1, attr1});
    auto subB0 = memref::SubViewOp::create(
        rewriter, loc, B, ArrayRef<OpFoldResult>{attr0, j},
        ArrayRef<OpFoldResult>{attr16, attr16},
        ArrayRef<OpFoldResult>{attr1, attr1});

    Value tokInitA = npu::DMALoad2DOp::create(
                         rewriter, loc, tokenTy, subA0, bank0A, c16, c16, c128)
                         .getToken();
    Value tokInitB = npu::DMALoad2DOp::create(
                         rewriter, loc, tokenTy, subB0, bank0B, c16, c16, c128)
                         .getToken();
    npu::DMAWaitOp::create(rewriter, loc, tokInitA);
    npu::DMAWaitOp::create(rewriter, loc, tokInitB);

    // Steady-state loop over nextK = 16..112: overlap next tile DMA load
    // with current bank's matmul compute.
    auto loopK = scf::ForOp::create(rewriter, loc, c16, c128, c16,
                                    ValueRange{c0});
    rewriter.setInsertionPointToStart(loopK.getBody());
    Value nextK = loopK.getInductionVar();
    Value curBank = loopK.getRegionIterArg(0);
    Value nextBank = arith::SubIOp::create(rewriter, loc, c1, curBank);

    Value nextBankA = getPingPongBankView(rewriter, loc, pingPongA, nextBank,
                                          attr0, attr16, attr1);
    Value nextBankB = getPingPongBankView(rewriter, loc, pingPongB, nextBank,
                                          attr0, attr16, attr1);
    auto subANext = memref::SubViewOp::create(
        rewriter, loc, A, ArrayRef<OpFoldResult>{i, OpFoldResult(nextK)},
        ArrayRef<OpFoldResult>{attr16, attr16},
        ArrayRef<OpFoldResult>{attr1, attr1});
    auto subBNext = memref::SubViewOp::create(
        rewriter, loc, B, ArrayRef<OpFoldResult>{OpFoldResult(nextK), j},
        ArrayRef<OpFoldResult>{attr16, attr16},
        ArrayRef<OpFoldResult>{attr1, attr1});

    Value tokNextA =
        npu::DMALoad2DOp::create(rewriter, loc, tokenTy, subANext, nextBankA,
                                 c16, c16, c128)
            .getToken();
    Value tokNextB =
        npu::DMALoad2DOp::create(rewriter, loc, tokenTy, subBNext, nextBankB,
                                 c16, c16, c128)
            .getToken();

    Value curBankA = getPingPongBankView(rewriter, loc, pingPongA, curBank,
                                         attr0, attr16, attr1);
    Value curBankB = getPingPongBankView(rewriter, loc, pingPongB, curBank,
                                         attr0, attr16, attr1);
    auto stepMatmul =
        cast<linalg::MatmulOp>(rewriter.clone(*matmulOp.getOperation()));
    stepMatmul.getDpsInputOperand(0)->set(curBankA);
    stepMatmul.getDpsInputOperand(1)->set(curBankB);
    stepMatmul.getDpsInitOperand(0)->set(sramC);

    npu::DMAWaitOp::create(rewriter, loc, tokNextA);
    npu::DMAWaitOp::create(rewriter, loc, tokNextB);
    scf::YieldOp::create(rewriter, loc, ValueRange{nextBank});

    rewriter.setInsertionPointAfter(loopK);

    // Epilogue: compute final tile (k = 112) sitting in finalBank.
    Value finalBank = loopK.getResult(0);
    Value finalBankA = getPingPongBankView(rewriter, loc, pingPongA, finalBank,
                                           attr0, attr16, attr1);
    Value finalBankB = getPingPongBankView(rewriter, loc, pingPongB, finalBank,
                                           attr0, attr16, attr1);
    auto finalMatmul =
        cast<linalg::MatmulOp>(rewriter.clone(*matmulOp.getOperation()));
    finalMatmul.getDpsInputOperand(0)->set(finalBankA);
    finalMatmul.getDpsInputOperand(1)->set(finalBankB);
    finalMatmul.getDpsInitOperand(0)->set(sramC);

    npu::DMAWaitOp::create(rewriter, loc, tokBias);

    auto newBias = cast<linalg::GenericOp>(rewriter.clone(*biasOp.getOperation()));
    newBias.getDpsInputOperand(0)->set(sramC);
    newBias.getDpsInputOperand(1)->set(sramBias);
    newBias.getDpsInitOperand(0)->set(sramC);

    if (reluOp)
      npu::VecReluOp::create(rewriter, loc, sramC);

    Value tokStore = npu::DMAStore2DOp::create(
                         rewriter, loc, tokenTy, sramC, subC, c16, c16, c128)
                         .getToken();
    npu::DMAWaitOp::create(rewriter, loc, tokStore);

    if (reluOp)
      rewriter.eraseOp(reluOp);
    rewriter.eraseOp(biasOp);
    rewriter.eraseOp(matmulOp);
  }
};
} // namespace

namespace mlir {
namespace npu {
std::unique_ptr<Pass> createFuseEpiloguePass() {
  return std::make_unique<FuseEpiloguePass>();
}

void registerFuseEpiloguePass() { PassRegistration<FuseEpiloguePass>(); }
} // namespace npu
} // namespace mlir
