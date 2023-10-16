use orion::operators::ml::{TreeRegressor, TreeRegressorTrait};
use orion::numbers::FixedTrait;


trait XGBoostPredictorTrait<T> {
    fn predict(trees: Span<TreeRegressor<T>>, features: Span<T>, weights: Span<T>) -> T;
}

fn predict<
    T,
    MAG,
    impl TFixed: FixedTrait<T, MAG>,
    impl TTreeRegressor: TreeRegressorTrait<T>,
    impl TMul: Mul<T>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    ref trees: Span<TreeRegressor<T>>, ref features: Span<T>, ref weights: Span<T>
) -> T {
    let mut sum_prediction: T = FixedTrait::ZERO();

    loop {
        match trees.pop_front() {
            Option::Some(tree) => {
                let mut tree = *tree;
                sum_prediction += tree.predict(features) * *weights.pop_front().unwrap()
            },
            Option::None(_) => {
                break;
            }
        };
    };

    sum_prediction
}
