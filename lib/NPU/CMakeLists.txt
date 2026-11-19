add_mlir_library(MLIRNPUDialect
  NPUDialect.cpp

  DEPENDS
  MLIRNPUOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
)
