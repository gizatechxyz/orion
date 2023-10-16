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
/// fit - Constructs a decision tree regressor based on the provided data and target values.
/// predict - Given a set of features, predicts the target value using the constructed decision tree.
trait TreeRegressorTrait<T> {
    /// # TreeRegressorTrait::fit
    ///
    /// ```rust 
    ///    fn fit(data: Span<Span<T>>, target: Span<T>, max_depth: usize, random_state: usize) -> TreeRegressor<T>;
    /// ```
    ///
    /// Builds a decision tree based on the provided data and target values up to a specified maximum depth.
    /// 
    /// ## Args
    ///
    /// * `data`: A span of spans representing rows of features in the dataset.
    /// * `target`: A span representing the target values corresponding to each row in the dataset.
    /// * `max_depth`: The maximum depth of the decision tree. The tree stops growing once this depth is reached.
    /// * `random_state`: It ensures that the tie-breaking is consistent across multiple runs, leading to reproducible results.
    ///
    /// ## Returns
    ///
    /// A `TreeRegressor` representing the root of the constructed decision tree.
    ///
    /// ## Type Constraints
    ///
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::operators::ml::{FP16x16TreeRegressor, TreeRegressorTrait};
    /// use orion::numbers::{FP16x16, FixedTrait};
    /// 
    /// fn tree_regressor_example() {
    ///
    ///  let data = array![
    ///      array![FixedTrait::new_unscaled(1, false), FixedTrait::new_unscaled(2, false)].span(),
    ///      array![FixedTrait::new_unscaled(3, false), FixedTrait::new_unscaled(4, false)].span(),
    ///      array![FixedTrait::new_unscaled(5, false), FixedTrait::new_unscaled(6, false)].span(),
    ///      array![FixedTrait::new_unscaled(7, false), FixedTrait::new_unscaled(8, false)].span(),
    ///  ]
    ///      .span();
    ///
    ///  let target = array![
    ///      FixedTrait::new_unscaled(2, false),
    ///      FixedTrait::new_unscaled(4, false),
    ///      FixedTrait::new_unscaled(6, false),
    ///      FixedTrait::new_unscaled(8, false)
    ///  ]
    ///      .span();
    ///
    ///  TreeRegressorTrait::fit(data, target, 3, 42);
    /// }
    /// ```
    ///
    fn fit(
        data: Span<Span<T>>, target: Span<T>, max_depth: usize, random_state: usize
    ) -> TreeRegressor<T>;
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
    /// Constrain input and output types to fixed point tensors.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::operators::ml::{FP16x16TreeRegressor, TreeRegressorTrait};
    /// use orion::numbers::{FP16x16, FixedTrait};
    /// 
    /// fn tree_regressor_example() {
    ///
    ///  let data = array![
    ///      array![FixedTrait::new_unscaled(1, false), FixedTrait::new_unscaled(2, false)].span(),
    ///      array![FixedTrait::new_unscaled(3, false), FixedTrait::new_unscaled(4, false)].span(),
    ///      array![FixedTrait::new_unscaled(5, false), FixedTrait::new_unscaled(6, false)].span(),
    ///      array![FixedTrait::new_unscaled(7, false), FixedTrait::new_unscaled(8, false)].span(),
    ///  ]
    ///      .span();
    ///
    ///  let target = array![
    ///      FixedTrait::new_unscaled(2, false),
    ///      FixedTrait::new_unscaled(4, false),
    ///      FixedTrait::new_unscaled(6, false),
    ///      FixedTrait::new_unscaled(8, false)
    ///  ]
    ///      .span();
    ///
    ///  let mut tree = TreeRegressorTrait::fit(data, target, 3);
    ///
    ///  let prediction_1 = tree
    ///    .predict(
    ///        array![FixedTrait::new_unscaled(1, false), FixedTrait::new_unscaled(2, false),].span()
    ///    );
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

fn mse<
    T,
    MAG,
    impl FFixedTrait: FixedTrait<T, MAG>,
    impl TSub: Sub<T>,
    impl TAddEq: AddEq<T>,
    impl TDiv: Div<T>,
    impl U32IntoMAG: Into<u32, MAG>,
    impl FeltTryIntoMAG: TryInto<felt252, MAG>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    y: Span<T>, prediction: T
) -> T {
    let mut sum_squared_error: T = FixedTrait::ZERO();

    let mut y_copy = y;
    loop {
        match y_copy.pop_front() {
            Option::Some(yi) => {
                let error = *yi - prediction;
                sum_squared_error += error
                    .pow(FixedTrait::new_unscaled(2.try_into().unwrap(), false));
            },
            Option::None(_) => {
                break;
            }
        };
    };

    sum_squared_error / FixedTrait::new_unscaled(y.len().into(), false)
}

fn best_split<
    T,
    MAG,
    impl FFixedTrait: FixedTrait<T, MAG>,
    impl TPartialOrd: PartialOrd<T>,
    impl TPartialEq: PartialEq<T>,
    impl TAddEq: AddEq<T>,
    impl TAdd: Add<T>,
    impl TSub: Sub<T>,
    impl TDiv: Div<T>,
    impl TMul: Mul<T>,
    impl U32IntoMAG: Into<u32, MAG>,
    impl FeltTryIntoMAG: TryInto<felt252, MAG>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    data: Span<Span<T>>, target: Span<T>, random_state: usize
) -> (usize, T, T) {
    let mut best_mse = FixedTrait::MAX();
    let mut best_split_feature = 0;
    let mut best_splits: Array<(usize, T, T)> = ArrayTrait::new();

    let n_features: u32 = (*data[0]).len();

    let mut feature = 0;
    loop {
        if feature == n_features {
            break;
        };

        let mut unique_values = ArrayTrait::new();
        let mut data_copy = data;
        loop {
            match data_copy.pop_front() {
                Option::Some(row) => {
                    unique_values.append(*row[feature])
                },
                Option::None(_) => {
                    break;
                }
            };
        };

        let mut unique_values = unique_values.span();
        loop {
            match unique_values.pop_front() {
                Option::Some(value) => {
                    let mut left_target = ArrayTrait::new();
                    let mut right_target = ArrayTrait::new();

                    let mut i = 0;
                    let mut target_copy = target;
                    loop {
                        match target_copy.pop_front() {
                            Option::Some(t) => {
                                if *(*data.at(i))[feature] < *value {
                                    left_target.append(*t);
                                } else {
                                    right_target.append(*t);
                                }
                                i += 1;
                            },
                            Option::None(_) => {
                                break;
                            }
                        };
                    };

                    if !left_target.is_empty() && !right_target.is_empty() {
                        let mut left_sum = FixedTrait::ZERO();
                        let mut left_target_copy = left_target.span();
                        loop {
                            match left_target_copy.pop_front() {
                                Option::Some(val) => {
                                    left_sum += *val;
                                },
                                Option::None(_) => {
                                    break;
                                }
                            };
                        };
                        let left_target_as_fp: T = FixedTrait::new_unscaled(
                            left_target.len().into(), false
                        );
                        let left_pred = left_sum / left_target_as_fp;

                        let mut right_sum = FixedTrait::ZERO();
                        let mut right_target_copy = right_target.span();
                        loop {
                            match right_target_copy.pop_front() {
                                Option::Some(val) => {
                                    right_sum += *val;
                                },
                                Option::None(_) => {
                                    break;
                                }
                            };
                        };
                        let right_target_as_fp: T = FixedTrait::new_unscaled(
                            right_target.len().into(), false
                        );
                        let right_pred = right_sum / right_target_as_fp;

                        let current_mse = (left_target_as_fp * mse(left_target.span(), left_pred))
                            + (right_target_as_fp * mse(right_target.span(), right_pred));

                        if !(current_mse > best_mse) {
                            if current_mse < best_mse {
                                best_mse = current_mse;
                                best_splits = array![];
                            }

                            let mut total_sum = FixedTrait::ZERO();
                            let mut target_copy = target;
                            loop {
                                match target_copy.pop_front() {
                                    Option::Some(t) => {
                                        total_sum += *t;
                                    },
                                    Option::None(_) => {
                                        break;
                                    }
                                };
                            };

                            let prediction = total_sum
                                / FixedTrait::new_unscaled(target.len().into(), false);

                            best_splits.append((feature, *value, prediction));
                        }
                    }
                },
                Option::None(_) => {
                    break;
                }
            };
        };

        feature += 1;
    };

    let random_idx: usize = u64_between(random_state.into(), 0, best_splits.len().into())
        .try_into()
        .unwrap();
    let (best_split_feature, best_split_value, best_prediction) = *best_splits.at(random_idx);

    (best_split_feature, best_split_value, best_prediction)
}

fn fit<
    T,
    MAG,
    impl FFixedTrait: FixedTrait<T, MAG>,
    impl TPartialOrd: PartialOrd<T>,
    impl TPartialEq: PartialEq<T>,
    impl TAddEq: AddEq<T>,
    impl TAdd: Add<T>,
    impl TSub: Sub<T>,
    impl TDiv: Div<T>,
    impl TMul: Mul<T>,
    impl U32IntoMAG: Into<u32, MAG>,
    impl FeltTryIntoMAG: TryInto<felt252, MAG>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    data: Span<Span<T>>, target: Span<T>, depth: usize, max_depth: usize, random_state: usize
) -> TreeRegressor<T> {
    if depth == max_depth || data.len() < 2 {
        let mut total = FixedTrait::ZERO();
        let mut target_copy = target;
        loop {
            match target_copy.pop_front() {
                Option::Some(val) => {
                    total += *val;
                },
                Option::None(_) => {
                    break;
                }
            };
        };
        return TreeRegressor {
            left: Option::None(()),
            right: Option::None(()),
            split_feature: 0,
            split_value: FixedTrait::ZERO(),
            prediction: total / FixedTrait::new_unscaled(target.len().into(), false),
        };
    }

    let (split_feature, split_value, prediction) = best_split(data, target, random_state);
    let mut left_data = ArrayTrait::new();
    let mut left_target = ArrayTrait::new();

    let mut right_data = ArrayTrait::new();
    let mut right_target = ArrayTrait::new();

    let mut data_copy = data;
    let mut i: usize = 0;
    loop {
        match data_copy.pop_front() {
            Option::Some(row) => {
                if *(*row).at(split_feature) < split_value {
                    left_data.append(row.clone());
                    left_target.append(*target[i])
                } else {
                    right_data.append(row.clone());
                    right_target.append(*target[i])
                }
                i += 1
            },
            Option::None(_) => {
                break;
            }
        };
    };

    TreeRegressor {
        left: Option::Some(
            BoxTrait::new(
                fit(left_data.span(), left_target.span(), depth + 1, max_depth, random_state)
            )
        ),
        right: Option::Some(
            BoxTrait::new(
                fit(right_data.span(), right_target.span(), depth + 1, max_depth, random_state)
            )
        ),
        split_feature,
        split_value,
        prediction,
    }
}
