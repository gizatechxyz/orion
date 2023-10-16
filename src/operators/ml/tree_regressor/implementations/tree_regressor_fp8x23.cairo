use orion::operators::ml::tree_regressor::core::{TreeRegressor, TreeRegressorTrait};
use orion::operators::ml::tree_regressor::core;
use orion::numbers::FP8x23;

impl FP8x23TreeRegressor of TreeRegressorTrait<FP8x23> {
    fn fit(
        data: Span<Span<FP8x23>>, target: Span<FP8x23>, max_depth: usize, random_state: usize
    ) -> TreeRegressor<FP8x23> {
        core::fit(data, target, 0, max_depth, random_state)
    }

    fn predict(ref self: TreeRegressor<FP8x23>, features: Span<FP8x23>) -> FP8x23 {
        core::predict(ref self, features)
    }
}
