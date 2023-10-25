use cubit::f64::procgen::rand::u64_between;

use orion::numbers::{FixedTrait};

#[derive(Copy, Drop)]
struct TreeRegressor<T> {
    left: Option<Box<TreeRegressor<T>>>,
    right: Option<Box<TreeRegressor<T>>>,
    split_feature: usize,
    split_value: T,
    prediction: T,
}

/// Trait
///
/// predict - Given a set of features, predicts the target value using the constructed decision tree.
trait TreeRegressorTrait<T> {
    /// # TreeRegressorTrait::predict
    ///
    /// ```rust 
    ///    fn predict(ref self: TreeRegressor<T>, features: Span<T>) -> T;
    /// ```
    ///
    /// Predicts the target value for a set of features using the provided decision tree.
    /// 
    /// ## Args
    ///
    /// * `self`: A reference to the decision tree used for making the prediction.
    /// * `features`: A span representing the features for which the prediction is to be made.
    ///
    /// ## Returns
    ///
    /// The predicted target value.
    ///
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::operators::ml::{FP16x16TreeRegressor, TreeRegressorTrait, TreeRegressor};
    /// use orion::numbers::{FP16x16, FixedTrait};
    /// 
    /// fn tree_regressor_example(tree: TreeRegressor<FP16x16>) {
    ///
    ///  tree.predict(
    ///        array![FixedTrait::new_unscaled(1, false), FixedTrait::new_unscaled(2, false),].span()
    ///    );
    ///    
    /// }
    /// ```
    ///
    fn predict(ref self: TreeRegressor<T>, features: Span<T>) -> T;
}

fn predict<
    T,
    MAG,
    impl FFixedTrait: FixedTrait<T, MAG>,
    impl TPartialOrd: PartialOrd<T>,
    impl FCopy: Copy<T>,
    impl FDrop: Drop<T>,
>(
    ref self: TreeRegressor<T>, features: Span<T>
) -> T {
    let mut current_node: TreeRegressor<T> = self;

    loop {
        match current_node.left {
            Option::Some(left) => {
                match current_node.right {
                    Option::Some(right) => {
                        if *features.at(current_node.split_feature) < current_node.split_value {
                            current_node = left.unbox();
                        } else {
                            current_node = right.unbox();
                        }
                    },
                    Option::None(_) => {
                        break;
                    }
                }
            },
            Option::None(_) => {
                break;
            }
        };
    };

    current_node.prediction
}
