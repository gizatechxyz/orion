use orion::operators::ml::tree_regressor::core::{TreeRegressor, TreeRegressorTrait};
use orion::operators::ml::tree_regressor::core;
use orion::numbers::FP16x16;

impl FP16x16TreeRegressor of TreeRegressorTrait<FP16x16> {
    fn fit(
        data: Span<Span<FP16x16>>, target: Span<FP16x16>, max_depth: usize, random_state: usize
    ) -> TreeRegressor<FP16x16> {
        core::fit(data, target, 0, max_depth, random_state)
    }

    fn predict(ref self: TreeRegressor<FP16x16>, features: Span<FP16x16>) -> FP16x16 {
        core::predict(ref self, features)
    }
}
