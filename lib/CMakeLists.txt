add_library(NPUPasses STATIC
  NPUPasses/LowerDMAToLLVM.cpp
  NPUPasses/PingPongDMAPipelining.cpp
  NPUPasses/TileMatmul.cpp
)

llvm_update_compile_flags(NPUPasses)

target_link_libraries(NPUPasses
  PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTilingInterface
  MLIRSupport
)
