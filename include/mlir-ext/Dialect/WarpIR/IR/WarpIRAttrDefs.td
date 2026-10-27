//===-- WarpIRAttrDefs.td - WarpIR attribute definitions -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines WarpIR attributes: memory space, data layout, buffer policy,
// pipe types, event IDs, and engine attributes. These are used as operands /
// attributes on WarpIR ops to encode explicit hardware control.
//
//===----------------------------------------------------------------------===//

#ifndef WARP_IR_ATTRDEFS
#define WARP_IR_ATTRDEFS

include "mlir-ext/Dialect/WarpIR/IR/WarpIRDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Attribute base class
//===----------------------------------------------------------------------===//

class WarpIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<WarpIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}


//===----------------------------------------------------------------------===//
// Warp distribution
//===----------------------------------------------------------------------===//
def Warp_DistributionAttr : WarpIR_Attr<"Distribution", "distributed"> {
  let summary = "Lane distribution of a warp-local tensor";
  let description = [{
    shape[d] == threadsPerDim[d] * elementsPerThread[d] * repeat[d]
    elementsPerThread[order[0]] contiguous elements are adjacent in memory
    (vectorizable).
  }];

  let parameters = (ins
    DefaultValuedParameter<"int64_t", "32">:$warpSize,
    ArrayRefParameter<"int64_t">:$threadsPerDim,
    ArrayRefParameter<"int64_t">:$elementsPerThread,
    ArrayRefParameter<"int64_t">:$repeat,
    ArrayRefParameter<"int64_t">:$order
  );

  let assemblyFormat = [{
    `<` `{`
      `warp_size` `=` $warpSize `,`
      `threads_per_dim` `=` $threadsPerDim `,`
      `elements_per_thread` `=` $elementsPerThread `,`
      `repeat` `=` $repeat `,`
      `order` `=` $order
    `}` `>`
  }];

  let genVerifyDecl = 1;
}
#endif // WARP_IR_ATTRDEFS