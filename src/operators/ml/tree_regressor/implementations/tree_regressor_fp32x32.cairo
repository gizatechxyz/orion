use orion::operators::ml::tree_regressor::core::{TreeNode, TreeRegressorTrait};
use orion::operators::ml::tree_regressor::core;
use orion::numbers::{FP32x32, FP32x32Impl};

impl FP32x32TreeRegressor of TreeRegressorTrait<FP32x32> {
    fn fit(
        data: Span<Span<FP32x32>>, target: Span<FP32x32>, max_depth: usize
    ) -> TreeNode<FP32x32> {
        core::fit(data, target, 0, max_depth)
    }

    fn predict(ref self: TreeNode<FP32x32>, features: Span<FP32x32>) -> FP32x32 {
        core::predict(ref self, features)
    }
}
