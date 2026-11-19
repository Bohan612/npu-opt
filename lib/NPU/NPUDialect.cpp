#include "NPU/NPUDialect.h"
#include "NPU/NPUOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::npu;

#include "NPU/NPUDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "NPU/NPUTypes.cpp.inc"

#define GET_OP_CLASSES
#include "NPU/NPUOps.cpp.inc"

void NPUDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "NPU/NPUTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "NPU/NPUOps.cpp.inc"
      >();
}
