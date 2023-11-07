use orion::operators::ml::tree_ensemble::tree_ensemble_classifier::{
    TreeEnsembleClassifierTrait, TreeEnsembleClassifier, predict
};
use orion::numbers::FP16x16;
use orion::operators::tensor::{FP16x16Tensor, U32Tensor, Tensor};

impl FP16x16TreeEnsembleClassifier of TreeEnsembleClassifierTrait<FP16x16> {
    fn predict(
        ref self: TreeEnsembleClassifier<FP16x16>, X: Tensor<FP16x16>
    ) -> (Tensor<usize>, Tensor<FP16x16>) {
        predict(ref self, X)
    }
}
