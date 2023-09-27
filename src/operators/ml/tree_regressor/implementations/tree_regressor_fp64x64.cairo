use orion::operators::ml::tree_regressor::core::{TreeNode, TreeRegressorTrait};
use orion::operators::ml::tree_regressor::core;
use orion::numbers::{FP64x64, FP64x64Impl};

impl FP64x64TreeRegressor of TreeRegressorTrait<FP64x64> {
    fn build_tree(
        data: Span<Span<FP64x64>>, target: Span<FP64x64>, depth: usize, max_depth: usize
    ) -> TreeNode<FP64x64> {
        core::build_tree(data, target, depth, max_depth)
    }

    fn predict(ref self: TreeNode<FP64x64>, features: Span<FP64x64>) -> FP64x64 {
        core::predict(ref self, features)
    }
}
