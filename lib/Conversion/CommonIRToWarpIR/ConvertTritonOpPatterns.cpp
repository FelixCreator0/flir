#include "mlir-ext/Conversion/CommonIRToWarpIR/Passes.h"
#include "mlir-ext/Dialect/WarpIR/IR/WarpIRDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"


// namespace {
// using namespace mlir;
// using namespace llvm;
// template <typename OpT>
// SmallVector<NamedAttribute> getTritonAttrs(OpT op) const {
//   SmallVector<NamedAttribute> attrs;
//   for (NamedAttribute attr : op->getAttrs()) {
//     if (attr.getName() == OpT::getOperandSegmentSizesAttrName(op->getName()))
//       continue;
//     attrs.push_back(attr);
//   }
//   return attrs;
// }

// } // namespace

namespace mlir {
namespace triton {
namespace warp {

using namespace mlir::triton;

class LoadOpConversionPattern : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern::OpConversionPattern;
  public:
  int numWarps;
  int threadsPerWarp;
  int numCTAs;
 
  LoadOpConversionPattern(TypeConverter &typeConverter, MLIRContext *context, int numWarps, int threadsPerWarp, int numCTAs)
      : OpConversionPattern<triton::LoadOp>(typeConverter, context), 
        numWarps(numWarps), threadsPerWarp(threadsPerWarp), numCTAs(numCTAs) {}
  
  SmallVector<NamedAttribute> getTritonAttrs(triton::LoadOp op) const{
    SmallVector<NamedAttribute> attrs;
    for (NamedAttribute attr : op->getAttrs()) {
      if (attr.getName() == op.getOperandSegmentSizesAttrName())
        continue;
      attrs.push_back(attr);
    }
    return attrs;
  }

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc(); 
    auto tensorType = mlir::dyn_cast<RankedTensorType>(op.getType());
    mlir::Type pointeeType = tensorType.getElementType();
    auto shape = tensorType.getShape();
    mlir::SmallVector<mlir::Attribute> shapeElements;
    for (int64_t dim : shape) {
    shapeElements.push_back(
        rewriter.getI64IntegerAttr(dim));
    }
    auto shapeAttr = rewriter.getArrayAttr(shapeElements);
    auto layout = inferDistribution(
        rewriter.getContext(),
        shape,
        numWarps,
        threadsPerWarp
    );
    auto resultType = warp::RegType::get(
        rewriter.getContext(),
        shape,
        pointeeType,
        layout
    );
    auto newOp = rewriter.create<warp::LoadOp>(
        loc,
        resultType,
        adaptor.getPtr(),
        getTritonAttrs(op)
    );
        
    rewriter.replaceOp(op, newOp.getResult());
    return success();
}
};

void populateTritonPatterns(RewritePatternSet &patterns,
                            TypeConverter &typeConverter, 
                            int numWarps, int threadsPerWarp, int numCTAs){
    MLIRContext *context = patterns.getContext();
    patterns.add<LoadOpConversionPattern>(typeConverter, context, numWarps, threadsPerWarp, numCTAs);
}
} // namespace warp
} // namespace triton
} // namespace mlir