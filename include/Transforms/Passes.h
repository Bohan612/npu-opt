#pragma once

#include <memory>

namespace mlir {
class Pass;
std::unique_ptr<Pass> createConvertMatmulToGPUPass();
} // namespace mlir
