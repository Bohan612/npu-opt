get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_executable(npu-opt
  npu-opt.cpp
)

llvm_update_compile_flags(npu-opt)
target_link_libraries(npu-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIROptLib
  NPUPasses
  MLIRNPUDialect
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
)
