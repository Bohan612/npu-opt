#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

struct TileTransfer {
  memref::AllocOp sramAlloc;
  memref::CopyOp copyOp;
  Value ddrSource;
  MemRefType tileType;
  int64_t numElements;
};

static int64_t getStaticNumElements(MemRefType type) {
  int64_t count = 1;
  for (int64_t dim : type.getShape()) {
    if (ShapedType::isDynamic(dim))
      return -1;
    count *= dim;
  }
  return count;
}

static Value createBufferSlotView(OpBuilder &b, Location loc, Value doubleBuf,
                                  Value slotIdx, MemRefType origTileType) {
  auto rank = origTileType.getRank();
  SmallVector<OpFoldResult> offsets(rank + 1, b.getIndexAttr(0));
  offsets[0] = slotIdx;

  SmallVector<OpFoldResult> sizes;
  sizes.push_back(b.getIndexAttr(1));
  for (int64_t dim : origTileType.getShape())
    sizes.push_back(b.getIndexAttr(dim));

  SmallVector<OpFoldResult> strides(rank + 1, b.getIndexAttr(1));

  auto subviewType = memref::SubViewOp::inferRankReducedResultType(
      origTileType.getShape(), cast<MemRefType>(doubleBuf.getType()), offsets,
      sizes, strides);

  return memref::SubViewOp::create(b, loc, cast<MemRefType>(subviewType),
                                   doubleBuf, offsets, sizes, strides);
}

static void emitAsyncDMAStart(OpBuilder &b, Location loc, Value srcMemRef,
                              Value dstSlotView, Value tagMemRef, Value slotIdx,
                              int64_t numElements) {
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value numElemsVal = arith::ConstantIndexOp::create(b, loc, numElements);

  auto srcType = cast<MemRefType>(srcMemRef.getType());
  auto dstType = cast<MemRefType>(dstSlotView.getType());

  SmallVector<Value> srcIndices(srcType.getRank(), c0);
  SmallVector<Value> dstIndices(dstType.getRank(), c0);
  SmallVector<Value> tagIndices = {slotIdx, c0};

  memref::DmaStartOp::create(b, loc, srcMemRef, srcIndices, dstSlotView,
                             dstIndices, numElemsVal, tagMemRef, tagIndices);
}

static void emitAsyncDMAWait(OpBuilder &b, Location loc, Value tagMemRef,
                             Value slotIdx, int64_t numElements) {
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value numElemsVal = arith::ConstantIndexOp::create(b, loc, numElements);
  SmallVector<Value> tagIndices = {slotIdx, c0};
  memref::DmaWaitOp::create(b, loc, tagMemRef, tagIndices, numElemsVal);
}

struct PingPongDMAPipeliningPass
    : public PassWrapper<PingPongDMAPipeliningPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PingPongDMAPipeliningPass)

  StringRef getArgument() const final { return "npu-ping-pong-dma"; }
  StringRef getDescription() const final {
    return "Promote SRAM buffers to double-buffers and software-pipeline "
           "scf.for loops with async DMA";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    scf::SCFDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    SmallVector<scf::ForOp> candidateLoops;
    funcOp.walk([&](scf::ForOp forOp) {
      if (forOp.getInitArgs().empty())
        candidateLoops.push_back(forOp);
    });

    for (scf::ForOp forOp : candidateLoops) {
      if (failed(pipelineLoop(forOp)))
        continue;
    }
  }

private:
  LogicalResult pipelineLoop(scf::ForOp forOp) {
    Block *body = forOp.getBody();
    SmallVector<TileTransfer> transfers;
    SmallVector<Operation *> addressOps;
    SmallVector<Operation *> computeOps;

    for (Operation &op : body->without_terminator()) {
      if (auto copyOp = dyn_cast<memref::CopyOp>(&op)) {
        auto dstAlloc = copyOp.getTarget().getDefiningOp<memref::AllocOp>();
        if (!dstAlloc)
          return failure();

        auto dstType = dstAlloc.getType();
        if (!dstType.hasStaticShape() || dstType.getMemorySpaceAsInt() != 1)
          return failure();

        if (dstAlloc->getBlock() == body)
          return failure();

        int64_t numElems = getStaticNumElements(dstType);
        if (numElems <= 0)
          return failure();

        transfers.push_back(
            {dstAlloc, copyOp, copyOp.getSource(), dstType, numElems});
      } else if (isa<memref::SubViewOp>(op) ||
                 op.getDialect()->getNamespace() == "arith") {
        addressOps.push_back(&op);
      } else {
        computeOps.push_back(&op);
      }
    }

    if (transfers.empty() || computeOps.empty())
      return failure();

    OpBuilder b(forOp);
    Location loc = forOp.getLoc();

    Value lb = forOp.getLowerBound();
    Value ub = forOp.getUpperBound();
    Value step = forOp.getStep();

    Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
    Value c2 = arith::ConstantIndexOp::create(b, loc, 2);
    Value steadyUb = arith::SubIOp::create(b, loc, ub, step);

    SmallVector<Value> doubleBuffers;
    SmallVector<Value> tagBuffers;
    auto tagMemRefType = MemRefType::get({2, 1}, b.getI32Type(),
                                         MemRefLayoutAttrInterface{},
                                         b.getI32IntegerAttr(1));

    for (TileTransfer &t : transfers) {
      OpBuilder allocBuilder(t.sramAlloc);
      SmallVector<int64_t> dbShape = {2};
      dbShape.append(t.tileType.getShape().begin(), t.tileType.getShape().end());

      auto dbType = MemRefType::get(dbShape, t.tileType.getElementType(),
                                    MemRefLayoutAttrInterface{},
                                    t.tileType.getMemorySpace());
      Value dbAlloc = memref::AllocOp::create(allocBuilder, loc, dbType);
      Value tagAlloc = memref::AllocOp::create(allocBuilder, loc, tagMemRefType);

      doubleBuffers.push_back(dbAlloc);
      tagBuffers.push_back(tagAlloc);
    }

    // Prologue: Prefetch Tile 0 into slot 0 (Ping)
    IRMapping prologueMap;
    prologueMap.map(forOp.getInductionVar(), lb);
    for (Operation *op : addressOps)
      b.clone(*op, prologueMap);

    for (size_t i = 0; i < transfers.size(); ++i) {
      Value src0 = prologueMap.lookupOrDefault(transfers[i].ddrSource);
      Value dst0 =
          createBufferSlotView(b, loc, doubleBuffers[i], c0, transfers[i].tileType);
      emitAsyncDMAStart(b, loc, src0, dst0, tagBuffers[i], c0,
                        transfers[i].numElements);
    }

    // Steady-State Loop: [lb, ub - step)
    auto newForOp = scf::ForOp::create(
        b, loc, lb, steadyUb, step, ValueRange{c0},
        [&](OpBuilder &loopB, Location loopLoc, Value iv, ValueRange iterArgs) {
          Value stage = iterArgs[0];
          Value curSlot = arith::RemUIOp::create(loopB, loopLoc, stage, c2);
          Value nextStage = arith::AddIOp::create(loopB, loopLoc, stage, c1);
          Value nextSlot = arith::RemUIOp::create(loopB, loopLoc, nextStage, c2);
          Value nextIv = arith::AddIOp::create(loopB, loopLoc, iv, step);

          IRMapping nextAddrMap;
          nextAddrMap.map(forOp.getInductionVar(), nextIv);
          for (Operation *op : addressOps)
            loopB.clone(*op, nextAddrMap);

          for (size_t i = 0; i < transfers.size(); ++i) {
            Value nextSrc = nextAddrMap.lookupOrDefault(transfers[i].ddrSource);
            Value nextDst = createBufferSlotView(
                loopB, loopLoc, doubleBuffers[i], nextSlot, transfers[i].tileType);
            emitAsyncDMAStart(loopB, loopLoc, nextSrc, nextDst, tagBuffers[i],
                              nextSlot, transfers[i].numElements);
          }

          IRMapping computeMap;
          computeMap.map(forOp.getInductionVar(), iv);
          for (size_t i = 0; i < transfers.size(); ++i) {
            emitAsyncDMAWait(loopB, loopLoc, tagBuffers[i], curSlot,
                             transfers[i].numElements);
            Value curView = createBufferSlotView(
                loopB, loopLoc, doubleBuffers[i], curSlot, transfers[i].tileType);
            computeMap.map(transfers[i].sramAlloc.getResult(), curView);
          }

          for (Operation *op : computeOps)
            loopB.clone(*op, computeMap);

          scf::YieldOp::create(loopB, loopLoc, ValueRange{nextStage});
        });

    // Epilogue: Wait and compute final tile
    Value finalStage = newForOp.getResult(0);
    Value finalSlot = arith::RemUIOp::create(b, loc, finalStage, c2);

    IRMapping epilogueMap;
    epilogueMap.map(forOp.getInductionVar(), steadyUb);
    for (size_t i = 0; i < transfers.size(); ++i) {
      emitAsyncDMAWait(b, loc, tagBuffers[i], finalSlot,
                       transfers[i].numElements);
      Value finalView = createBufferSlotView(
          b, loc, doubleBuffers[i], finalSlot, transfers[i].tileType);
      epilogueMap.map(transfers[i].sramAlloc.getResult(), finalView);
    }

    for (Operation *op : computeOps)
      b.clone(*op, epilogueMap);

    forOp.erase();
    for (TileTransfer &t : transfers) {
      if (t.sramAlloc.use_empty())
        t.sramAlloc.erase();
    }

    return success();
  }
};

} // namespace

namespace mlir::npu {
std::unique_ptr<Pass> createPingPongDMAPipeliningPass() {
  return std::make_unique<PingPongDMAPipeliningPass>();
}
} // namespace mlir::npu
