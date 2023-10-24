use orion::operators::ml::tree_classifier::core::{TreeClassifier, TreeClassifierTrait};
use orion::operators::ml::tree_classifier::core;
use orion::numbers::{FP64x64, FP64x64Impl};

impl FP64x64TreeClassifier of TreeClassifierTrait<FP64x64> {
    fn predict(ref self: TreeClassifier<FP64x64>, features: Span<FP64x64>) -> FP64x64 {
        core::predict(ref self, features)
    }

    fn predict_proba(ref self: TreeClassifier<FP64x64>, features: Span<FP64x64>) -> Span<FP64x64> {
        core::predict_proba(ref self, features)
    }
}
