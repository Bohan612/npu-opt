#ifndef NPU_OPS_TD
#define NPU_OPS_TD

include "NPU/NPUDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class NPU_Op<string mnemonic, list<Trait> traits = []>
    : Op<NPU_Dialect, mnemonic, traits>;

def NPU_DMALoadOp : NPU_Op<"dma_load"> {
  let summary = "Asynchronously transfer a tile from DDR (space 0) to SRAM (space 1)";
  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst);
  let results = (outs NPU_TokenType:$token);
  let assemblyFormat = "$src `to` $dst attr-dict `:` type($src) `to` type($dst) `->` type($token)";
}

def NPU_DMAWaitOp : NPU_Op<"dma_wait"> {
  let summary = "Block compute pipeline until async DMA token completes";
  let arguments = (ins NPU_TokenType:$token);
  let assemblyFormat = "$token attr-dict `:` type($token)";
}

def NPU_MMAOp : NPU_Op<"mma", [Pure, AllTypesMatch<["acc", "res"]>]> {
  let summary = "16x16x16 systolic Cube matrix-multiply-accumulate";
  let arguments = (ins
    VectorOfAnyRankOf<[F16]>:$lhs,
    VectorOfAnyRankOf<[F16]>:$rhs,
    VectorOfAnyRankOf<[F16]>:$acc
  );
  let results = (outs VectorOfAnyRankOf<[F16]>:$res);
  let assemblyFormat = "$lhs `,` $rhs `,` $acc attr-dict `:` type($lhs) `,` type($rhs) `->` type($res)";
}

#endif // NPU_OPS_TD

def NPU_VecAddOp : NPU_Op<"vec_add", []> {
  let summary = "NPU 16x16 vector-bias addition";
  let description = [{
    Executes a 16x16 vector addition in the NPU's Vector ALU. 
    Adds a 1x16 bias tile to a 16x16 matrix tile currently in SRAM.
  }];
  let arguments = (ins
    Arg<AnyMemRef, "the 16x16 SRAM matrix tile (in/out)", [MemWrite]>:$matrix_tile,
    Arg<AnyMemRef, "the 1x16 SRAM bias tile", [MemRead]>:$bias_tile
  );
  let assemblyFormat = "$matrix_tile `,` $bias_tile attr-dict `:` type($matrix_tile) `,` type($bias_tile)";
}
