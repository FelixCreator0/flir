# include "mlir-ext/Conversion/CommonIRToWarpIR/InferWarpDistribution.h"

namespace mlir::triton::warp {
using namespace llvm;
static DistributionAttr getDefaultWarpLayout(MLIRContext *ctx,
                                             ArrayRef<int64_t> shape,
                                             int64_t warpSize = 32) {
  int rank = shape.size();
  SmallVector<int64_t> order(rank), threads(rank, 1), elems(rank, 1),
      repeat(shape.begin(), shape.end());
  for (int i = 0; i < rank; ++i) order[i] = rank - 1 - i;   // 最快维在前

  int64_t fast = order[0];
  if (shape[fast] % 4 == 0) {
    elems[fast] = 4;
    repeat[fast] = shape[fast] / 4;
  }

  int64_t remaining = warpSize;
  for (int64_t d : order) {
    int64_t t = std::gcd(remaining, repeat[d]);
    threads[d] = t;
    repeat[d] /= t;
    remaining /= t;
  }
  if (remaining != 1)            // tile 太小,v1 不支持 lane 复制
    return {};
  return DistributionAttr::get(ctx, warpSize, threads, elems, repeat, order);
}

DistributionAttr inferDistribution(
    mlir::MLIRContext *ctx,
    const llvm::ArrayRef<int64_t> shape,
    int64_t numWarps,
    int64_t warpSize){
    
    return getDefaultWarpLayout(ctx, shape, warpSize);
    
//     auto threadsPerDim = SmallVector<int64_t>{4, 8};
//     auto elementsPerThread = SmallVector<int64_t>{1, 4};
//     auto order = SmallVector<int64_t>{0, 1};
//     auto repeat = SmallVector<int64_t>{2, 1};
//     return DistributionAttr::get(
//             ctx,
//             /*warpSize*/warpSize,
//             /*threadsPerDim*/threadsPerDim,
//             /*elementsPerThread*/elementsPerThread,
//             /*repeat*/repeat,
//             /*order*/order);

}

}