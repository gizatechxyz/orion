use orion::operators::ml::tree_classifier::core::{TreeNode, TreeClassifierTrait};
use orion::operators::ml::tree_classifier::core;
use orion::numbers::FP8x23;

impl FP8x23TreeClassifier of TreeClassifierTrait<FP8x23> {
    fn predict(ref self: TreeNode<FP8x23>, features: Span<FP8x23>) -> FP8x23 {
        core::predict(ref self, features)
    }

    fn predict_proba(ref self: TreeNode<FP8x23>, features: Span<FP8x23>) -> Span<FP8x23> {
        core::predict_proba(ref self, features)
    }
}
