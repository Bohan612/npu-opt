set(LLVM_LINK_COMPONENTS
  Support
)

set(LIBS
  GPUTransforms
  NPUPasses
  MLIRNPUDialect
  MLIRArithDialect
  MLIRArithToLLVM
  MLIRControlFlowDialect
  MLIRControlFlowToLLVM
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRGPUTransforms
  MLIRGPUToROCDLTransforms
  MLIRGPUROCDLTransforms
  MLIRLinalgDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIROptLib
  MLIRLinalgTransforms
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRROCDLDialect
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRTransforms
  MLIRUBToLLVM
  MLIRTransformUtils
)

add_llvm_executable(npu-opt
  npu-opt.cpp
)

llvm_update_compile_flags(npu-opt)
target_link_libraries(npu-opt PRIVATE ${LIBS}  MLIRComplexToLLVM
  MLIRIndexToLLVM
  MLIRMPIToLLVM
  MLIRNVVMToLLVM
  MLIROpenMPToLLVMConversion
  MLIRPtrToLLVM
)
mlir_check_all_link_libraries(npu-opt)
