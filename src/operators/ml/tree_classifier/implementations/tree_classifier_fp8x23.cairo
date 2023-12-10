use orion::operators::ml::tree_classifier::core::{
    TreeClassifier, TreeClassifierTrait, predict, predict_proba
};
use orion::operators::ml::tree_classifier::core;
use orion::numbers::FP8x23;

impl FP8x23TreeClassifier of TreeClassifierTrait<FP8x23> {
    fn predict(ref self: TreeClassifier<FP8x23>, features: Span<FP8x23>) -> FP8x23 {
        predict(ref self, features)
    }

    fn predict_proba(ref self: TreeClassifier<FP8x23>, features: Span<FP8x23>) -> Span<FP8x23> {
        predict_proba(ref self, features)
    }
}
