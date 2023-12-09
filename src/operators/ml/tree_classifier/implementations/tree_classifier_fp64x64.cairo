use orion::operators::ml::tree_classifier::core::{TreeClassifier, TreeClassifierTrait, predict, predict_proba};
use orion::operators::ml::tree_classifier::core;
use orion::numbers::{FP64x64, FP64x64Impl};

impl FP64x64TreeClassifier of TreeClassifierTrait<FP64x64> {
    fn predict(ref self: TreeClassifier<FP64x64>, features: Span<FP64x64>) -> FP64x64 {
        predict(ref self, features)
    }

    fn predict_proba(ref self: TreeClassifier<FP64x64>, features: Span<FP64x64>) -> Span<FP64x64> {
        predict_proba(ref self, features)
    }
}
