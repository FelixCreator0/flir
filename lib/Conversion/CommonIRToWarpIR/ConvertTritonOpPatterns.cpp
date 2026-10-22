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

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "Converting LoadOp: " << op << "\n";
    return success();
    }
};

void populateTritonPatterns(RewritePatternSet &patterns,
                                    TypeConverter &typeConverter){
    MLIRContext *context = patterns.getContext();
    patterns.add<LoadOpConversionPattern>(typeConverter, context);
}

} // namespace warp
} // namespace triton
} // namespace mlir