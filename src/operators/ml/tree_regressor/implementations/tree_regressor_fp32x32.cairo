use orion::operators::ml::tree_regressor::core::{TreeNode, TreeRegressorTrait};
use orion::operators::ml::tree_regressor::core;
use orion::numbers::{FP32x32, FP32x32Impl};

impl FP32x32TreeRegressor of TreeRegressorTrait<FP32x32> {
    fn predict(ref self: TreeNode<FP32x32>, features: Span<FP32x32>) -> FP32x32 {
        core::predict(ref self, features)
    }
}
