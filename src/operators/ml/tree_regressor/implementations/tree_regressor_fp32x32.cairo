use orion::operators::ml::tree_regressor::core::{TreeRegressor, TreeRegressorTrait};
use orion::operators::ml::tree_regressor::core;
use orion::numbers::{FP32x32, FP32x32Impl};

impl FP32x32TreeRegressor of TreeRegressorTrait<FP32x32> {
    fn fit(
        data: Span<Span<FP32x32>>, target: Span<FP32x32>, max_depth: usize, random_state: usize
    ) -> TreeRegressor<FP32x32> {
        core::fit(data, target, 0, max_depth, random_state)
    }

    fn predict(ref self: TreeRegressor<FP32x32>, features: Span<FP32x32>) -> FP32x32 {
        core::predict(ref self, features)
    }
}
