use orion::operators::ml::tree_regressor::core::{TreeNode, TreeRegressorTrait};
use orion::operators::ml::tree_regressor::core;
use orion::numbers::FP16x16;

impl FP16x16TreeRegressor of TreeRegressorTrait<FP16x16> {
    fn predict(ref self: TreeNode<FP16x16>, features: Span<FP16x16>) -> FP16x16 {
        core::predict(ref self, features)
    }
}
