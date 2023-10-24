use orion::operators::ml::tree_regressor::core::{TreeRegressor, TreeRegressorTrait};
use orion::operators::ml::tree_regressor::core;
use orion::numbers::FP8x23;

impl FP8x23TreeRegressor of TreeRegressorTrait<FP8x23> {
    fn predict(ref self: TreeRegressor<FP8x23>, features: Span<FP8x23>) -> FP8x23 {
        core::predict(ref self, features)
    }
}
