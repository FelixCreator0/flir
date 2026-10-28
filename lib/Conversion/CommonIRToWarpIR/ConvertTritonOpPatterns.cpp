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
  
  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto resultTy = getTypeConverter()->convertType(op.getResult().getType());
    auto tensorTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    auto enc = inferDistribution(this->getContext(), tensorTy.getShape(), this->numWarps, this->threadsPerWarp);
    auto newOp = rewriter.create<warp::LoadOp>(
        op.getLoc(), TypeRange{resultTy}, 
        adaptor.getPtr(), adaptor.getMask(), adaptor.getOther(),
        op.getBoundaryCheckAttr(), op.getPaddingAttr(), op.getCacheAttr(),
        op.getEvictAttr(), op.getIsVolatileAttr(), op.getFlagtreeHintsAttr(), enc
    );

    newOp->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    rewriter.replaceOp(op, newOp->getResults());
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