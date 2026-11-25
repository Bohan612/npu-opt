add_library(NPUPasses STATIC
  NPUPasses/VectorizeMatmul.cpp
  NPUPasses/ConvertToNPUDialect.cpp
  NPUPasses/ConvertNPUToLLVM.cpp
  NPUPasses/LowerDMAToLLVM.cpp
  NPUPasses/PingPongDMAPipelining.cpp
  NPUPasses/TileMatmul.cpp
  NPUPasses/FuseEpilogue.cpp
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
  MLIRMemRefTransforms
  MLIRAffineToStandard
  MLIRVectorToLLVMPass
  MLIRVectorToSCF
  MLIRArithToLLVM
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTilingInterface
  MLIRSupport
)

add_subdirectory(NPU)
add_subdirectory(Transforms)
