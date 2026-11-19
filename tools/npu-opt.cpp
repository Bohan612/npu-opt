#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "NPU/NPUDialect.h"
#include "NPU/NPUOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace mlir::npu {
std::unique_ptr<mlir::Pass> createConvertToNPUDialectPass();
std::unique_ptr<mlir::Pass> createConvertNPUToLLVMPass();
std::unique_ptr<mlir::Pass> createVectorizeMatmulPass(bool lowerContract = true);
std::unique_ptr<mlir::Pass> createLowerDMAToLLVMPass();
std::unique_ptr<mlir::Pass> createPingPongDMAPipeliningPass();
std::unique_ptr<mlir::Pass> createTileMatmulPass();
} // namespace mlir::npu

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::npu::NPUDialect>();
  mlir::registerAllPasses();

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::npu::createLowerDMAToLLVMPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::npu::createPingPongDMAPipeliningPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::npu::createTileMatmulPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::npu::createVectorizeMatmulPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::npu::createConvertToNPUDialectPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::npu::createConvertNPUToLLVMPass();
  });

  
  mlir::PassPipelineRegistration<>(
      "npu-compile-pipeline",
      "End-to-end NPU compilation pipeline (Linalg -> SRAM Ping-Pong DMA -> NPU Dialect -> LLVM)",
      [](mlir::OpPassManager &pm) {
        pm.addNestedPass<mlir::func::FuncOp>(mlir::npu::createTileMatmulPass());
        pm.addNestedPass<mlir::func::FuncOp>(mlir::npu::createPingPongDMAPipeliningPass());
        pm.addNestedPass<mlir::func::FuncOp>(mlir::npu::createVectorizeMatmulPass(false));
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addNestedPass<mlir::func::FuncOp>(mlir::npu::createConvertToNPUDialectPass());
        pm.addPass(mlir::memref::createExpandStridedMetadataPass());
        pm.addPass(mlir::createLowerAffinePass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createConvertVectorToSCFPass());
        pm.addPass(mlir::createSCFToControlFlowPass());
        pm.addPass(mlir::createConvertVectorToLLVMPass());
        pm.addPass(mlir::npu::createConvertNPUToLLVMPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
      });

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "NPU Out-of-Tree Optimizer Driver\n", registry));
}
