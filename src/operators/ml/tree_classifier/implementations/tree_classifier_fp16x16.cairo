use orion::operators::ml::tree_classifier::core::{
    TreeClassifier, TreeClassifierTrait, predict, predict_proba
};
use orion::operators::ml::tree_classifier::core;
use orion::numbers::FP16x16;

impl FP16x16TreeClassifier of TreeClassifierTrait<FP16x16> {
    fn predict(ref self: TreeClassifier<FP16x16>, features: Span<FP16x16>) -> FP16x16 {
        predict(ref self, features)
    }

    fn predict_proba(ref self: TreeClassifier<FP16x16>, features: Span<FP16x16>) -> Span<FP16x16> {
        predict_proba(ref self, features)
    }
}
