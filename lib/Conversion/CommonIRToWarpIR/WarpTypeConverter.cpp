#include "mlir-ext/Conversion/CommonIRToWarpIR/WarpTypeConverter.h"
#include <algorithm>
#include <numeric>
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir-ext/Conversion/CommonIRToWarpIR/Passes.h"
#include "mlir-ext/Dialect/WarpIR/IR/WarpIRDialect.h"
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"


using namespace mlir;
using namespace mlir::triton::warp;
using namespace mlir::triton::tile;

//
// TypeConverter
//
WarpTypeConverter::WarpTypeConverter(MLIRContext *context,
                                     int numWarps, int threadsPerWarp,
                                     int numCTAs)
    : context(context), numWarps(numWarps), threadsPerWarp(threadsPerWarp),
      numCTAs(numCTAs) {
  addConversion([](Type type) { return type; });

  addConversion([this](RankedTensorType tensorType)-> Type {
    auto elementType = tensorType.getElementType();

    // 处理 tensor<...x!tt.ptr<...>>
    if (auto ptrType = dyn_cast<triton::PointerType>(elementType)) {
        auto pointeeType = ptrType.getPointeeType();

        auto memorySpace = IntegerAttr::get(
            IntegerType::get(this->context, 32),
            ptrType.getAddressSpace());

        auto warpPtrType = warp::PtrType::get(
            this->context, pointeeType, memorySpace);

        return RankedTensorType::get(
            tensorType.getShape(), warpPtrType,
            tensorType.getEncoding());
    }

    return tensorType;
 });

  addConversion([this](RankedTensorType tensorType)-> Type {
    auto elementType = tensorType.getElementType();

    // handle tensor<...x!tt.ptr<...>>
    if (auto ptrType = dyn_cast<triton::PointerType>(elementType)) {
        auto pointeeType = ptrType.getPointeeType();

        auto memorySpace = IntegerAttr::get(
            IntegerType::get(this->context, 32),
            ptrType.getAddressSpace());

        auto warpPtrType = warp::PtrType::get(
            this->context, pointeeType, memorySpace);

        return RankedTensorType::get(
            tensorType.getShape(), warpPtrType,
            tensorType.getEncoding());
    }

    return tensorType;
 });
//   addConversion([this](triton::PointerType ptrType) -> warp::PtrType {
//     auto pointeeTensorType =
//         dyn_cast<RankedTensorType>(ptrType.getPointeeType());

//     mlir::Attribute memorySpace =
//     mlir::IntegerAttr::get(
//         mlir::IntegerType::get(this->context, 32),
//         ptrType.getAddressSpace());

//     return warp::PtrType::get(
//         this->context,
//         pointeeTensorType,
//         memorySpace
//     );

//   });

  addSourceMaterialization(
    [](mlir::OpBuilder &builder,
       mlir::Type resultType,
       mlir::ValueRange inputs,
       mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1 ||
          !mlir::isa<warp::RegType>(inputs.front().getType()) ||
          !mlir::isa<mlir::RankedTensorType>(resultType))
        return {};

      auto op = builder.create<warp::RegToTensorOp>(
          loc, resultType, inputs.front());

      return op.getResult();
    });

    // addTargetMaterialization(
    // [](OpBuilder &builder,
    //    Type resultType,
    //    ValueRange inputs,
    //    Location loc) -> Value {
    //   if (inputs.size() != 1)
    //     return {};

    //   auto sourceType = dyn_cast<RankedTensorType>(
    //       inputs.front().getType());
    //   auto targetType = dyn_cast<RankedTensorType>(resultType);

    //   if (!sourceType || !targetType)
    //     return {};

    //   auto sourcePtrType =
    //       dyn_cast<triton::PointerType>(sourceType.getElementType());
    //   auto targetPtrType =
    //       dyn_cast<warp::PtrType>(targetType.getElementType());

    //   if (!sourcePtrType || !targetPtrType)
    //     return {};

    //   return builder
    //       .create<warp::PtrTensorCastOp>(
    //           loc, resultType, inputs.front())
    //       .getResult();
    // });

}
