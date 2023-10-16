use orion::operators::ml::xgboost_regressor::core::{TreeRegressor, XGBoostRegressorTrait};
use orion::operators::ml::xgboost_regressor::core;
use orion::operators::ml::FP8x23TreeRegressor;
use orion::numbers::FP8x23;

impl FP8x23XGBoostRegressor of XGBoostRegressorTrait<FP8x23> {
    fn predict(
        ref self: Span<TreeRegressor<FP8x23>>, ref features: Span<FP8x23>, ref weights: Span<FP8x23>
    ) -> FP8x23 {
        core::predict(ref self, ref features, ref weights)
    }
}
