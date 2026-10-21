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
namespace warp{
#define GEN_PASS_DEF_COMMONIRTOWARPIR
#include "mlir-ext/Conversion/CommonIRToWarpIR/Passes.h.inc"
} // namespace warp
} // namespace triton
} // namespace mlir

#define DEBUG_TYPE "commonir-to-warpir-conversion"

namespace mlir{
namespace triton{
namespace warp{

void populateAddWarpConfigsPatterns(RewritePatternSet &patterns,
                                    TypeConverter &typeConverter){

}

class CommonIRToWarpIRPass
    : public impl::CommonIRToWarpIRBase<CommonIRToWarpIRPass> {
  using CommonIRToWarpIRBase<
      CommonIRToWarpIRPass>::CommonIRToWarpIRBase;

    void runOnOperation() override {
        auto moduleOp = getOperation();
        mlir::OpBuilder builder(moduleOp);
        mlir::RewritePatternSet patterns(&getContext());
        mlir::TypeConverter typeConverter;
        MLIRContext *ctx = &getContext();
        auto shape = SmallVector<int64_t>{16, 32};
        auto warpDistribution = inferDistribution(ctx, shape, numWarps, threadsPerWarp);

        llvm::errs() << "Distribution: " << warpDistribution << "\n";
        populateAddWarpConfigsPatterns(patterns, typeConverter);

        if (mlir::failed(mlir::applyPatternsGreedily(
            moduleOp, std::move(patterns)))) {
                signalPassFailure();
                return;
        }
    }
};


} // namespace warp
} // namespace triton
} // namespace mlir
