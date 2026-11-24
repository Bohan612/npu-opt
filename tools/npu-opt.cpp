#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Conversion/XeVMToLLVM/XeVMToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/PtrToLLVM/PtrToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/MPIToLLVM/MPIToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "Transforms/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// NPU Pass Declarations
namespace mlir::npu {
  std::unique_ptr<mlir::Pass> createTileMatmulPass();
  std::unique_ptr<mlir::Pass> createPingPongDMAPipeliningPass();
  std::unique_ptr<mlir::Pass> createVectorizeMatmulPass(bool);
  std::unique_ptr<mlir::Pass> createConvertToNPUDialectPass();
  std::unique_ptr<mlir::Pass> createConvertNPUToLLVMPass();
}

static void registerNPUMatmulPipeline() {
  static mlir::PassPipelineRegistration<> reg(
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
}
// 
#include "NPU/NPUDialect.h"// #include "NPU/NPUOps.h"// namespace mlir::npu {// std::unique_ptr<mlir::Pass> createConvertToNPUDialectPass();// std::unique_ptr<mlir::Pass> createConvertNPUToLLVMPass();// } // namespace mlir::npu//   registry.insert<mlir::npu::NPUDialect>();//       "npu-compile-pipeline",//       "End-to-end NPU compilation pipeline (Linalg -> SRAM Ping-Pong DMA -> NPU Dialect -> LLVM)",//         pm.addNestedPass<mlir::func::FuncOp>(mlir::npu::createTileMatmulPass());//         pm.addNestedPass<mlir::func::FuncOp>(mlir::npu::createPingPongDMAPipeliningPass());//         pm.addNestedPass<mlir::func::FuncOp>(mlir::npu::createVectorizeMatmulPass(false));//         pm.addNestedPass<mlir::func::FuncOp>(mlir::npu::createConvertToNPUDialectPass());//         pm.addPass(mlir::npu::createConvertNPUToLLVMPass());
//       mlir::MlirOptMain(argc, argv, "NPU & GPU Optimizer Driver\n", registry));

static void registerGPUMatmulPipeline() {
  static mlir::PassPipelineRegistration<> reg(
      "gpu-matmul-pipeline",
      "Lower tiled matmul to GPU dialect (workgroup shared memory + barriers) and ROCDL/LLVM",
      [](mlir::OpPassManager &pm) {
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertMatmulToGPUPass());
        pm.addPass(mlir::createGpuKernelOutliningPass());

        auto &gpuPm = pm.nest<mlir::gpu::GPUModuleOp>();
        gpuPm.addNestedPass<mlir::gpu::GPUFuncOp>(mlir::createSCFToControlFlowPass());
        gpuPm.addPass(mlir::createConvertGpuOpsToROCDLOps(mlir::ConvertGpuOpsToROCDLOpsOptions{.arch = "gfx90a"}));
        gpuPm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        gpuPm.addPass(mlir::createArithToLLVMConversionPass());
        gpuPm.addPass(mlir::createConvertControlFlowToLLVMPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
      });
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::registerConvertComplexToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::gpu::registerConvertGpuToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::mpi::registerConvertMPIToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerConvertNVVMToLLVMInterface(registry);
  mlir::registerConvertOpenMPToLLVMInterface(registry);
  mlir::ptr::registerConvertPtrToLLVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::vector::registerConvertVectorToLLVMInterface(registry);
  registry.insert<mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect,
                  mlir::func::FuncDialect,
                  mlir::gpu::GPUDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::ROCDL::ROCDLDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect,
                  mlir::affine::AffineDialect,
                  mlir::vector::VectorDialect,
                  mlir::npu::NPUDialect>();

  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  registerGPUMatmulPipeline();
  registerNPUMatmulPipeline();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "NPU & GPU Optimizer Driver\n", registry));
}
