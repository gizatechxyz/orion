use orion::operators::ml::tree_classifier::core::{
    TreeClassifier, TreeClassifierTrait, predict, predict_proba
};
use orion::operators::ml::tree_classifier::core;
use orion::numbers::{FP32x32, FP32x32Impl};

impl FP32x32TreeClassifier of TreeClassifierTrait<FP32x32> {
    fn predict(ref self: TreeClassifier<FP32x32>, features: Span<FP32x32>) -> FP32x32 {
        predict(ref self, features)
    }

    fn predict_proba(ref self: TreeClassifier<FP32x32>, features: Span<FP32x32>) -> Span<FP32x32> {
        predict_proba(ref self, features)
    }
}
