add_library(NPUPasses STATIC
  NPUPasses/PingPongDMAPipelining.cpp
)

llvm_update_compile_flags(NPUPasses)

target_link_libraries(NPUPasses
  PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
)
