use orion::operators::ml::tree_regressor::core::{TreeRegressor, TreeRegressorTrait};
use orion::operators::ml::tree_regressor::core;
use orion::numbers::{FP64x64, FP64x64Impl};

impl FP64x64TreeRegressor of TreeRegressorTrait<FP64x64> {
    fn predict(ref self: TreeRegressor<FP64x64>, features: Span<FP64x64>) -> FP64x64 {
        core::predict(ref self, features)
    }
}
