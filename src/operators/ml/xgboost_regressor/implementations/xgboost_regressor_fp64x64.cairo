use orion::operators::ml::xgboost_regressor::core::{TreeRegressor, XGBoostRegressorTrait};
use orion::operators::ml::xgboost_regressor::core;
use orion::operators::ml::FP64x64TreeRegressor;
use orion::numbers::{FP64x64, FP64x64Impl};

impl FP64x64XGBoostRegressor of XGBoostRegressorTrait<FP64x64> {
    fn predict(
        ref self: Span<TreeRegressor<FP64x64>>, ref features: Span<FP64x64>, ref weights: Span<FP64x64>
    ) -> FP64x64 {
        core::predict(ref self, ref features, ref weights)
    }
}
