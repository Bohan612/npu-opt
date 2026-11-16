#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace mlir::npu {
std::unique_ptr<mlir::Pass> createPingPongDMAPipeliningPass();
std::unique_ptr<mlir::Pass> createTileMatmulPass();
} // namespace mlir::npu

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::npu::createPingPongDMAPipeliningPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::npu::createTileMatmulPass();
  });

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "NPU Out-of-Tree Optimizer Driver\n", registry));
}
