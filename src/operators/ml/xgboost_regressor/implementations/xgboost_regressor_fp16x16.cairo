use orion::operators::ml::xgboost_regressor::core::{TreeNode, XGBoostRegressorTrait};
use orion::operators::ml::xgboost_regressor::core;
use orion::operators::ml::FP16x16TreeRegressor;
use orion::numbers::FP16x16;

impl FP16x16XGBoostRegressor of XGBoostRegressorTrait<FP16x16> {
    fn predict(
        ref self: Span<TreeNode<FP16x16>>, ref features: Span<FP16x16>, ref weights: Span<FP16x16>
    ) -> FP16x16 {
        core::predict(ref self, ref features, ref weights)
    }
}
