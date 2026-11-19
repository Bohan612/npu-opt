#ifndef NPU_DIALECT_TD
#define NPU_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def NPU_Dialect : Dialect {
  let name = "npu";
  let summary = "Hardware-level NPU dialect for async DMA and systolic MMA";
  let cppNamespace = "::mlir::npu";
  let useDefaultTypePrinterParser = 1;
}

class NPU_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<NPU_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def NPU_TokenType : NPU_Type<"Token", "token"> {
  let summary = "Asynchronous DMA synchronization token";
}

#endif // NPU_DIALECT_TD
