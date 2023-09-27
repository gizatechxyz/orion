use array::{IndexView, SpanTrait, ArrayTrait};

use orion::numbers::{FP16x16, FixedTrait, FP16x16Impl};
use orion::numbers::fixed_point::implementations::fp16x16::core::MAX;
use orion::operators::tensor::{Tensor, FP16x16Tensor};

#[derive(Copy, Drop)]
struct TreeNode {
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
    split_feature: usize,
    split_value: FP16x16,
    prediction: FP16x16,
}

#[generate_trait]
impl TreeNodeImpl of TreeNodeTrait {
    fn predict(ref self: TreeNode, features: Span<FP16x16>) -> FP16x16 {
        let mut current_node: TreeNode = self;

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
}

fn mse(y: Span<FP16x16>, prediction: FP16x16) -> FP16x16 {
    let mut sum_squared_error: FP16x16 = FixedTrait::new(0, false);

    let mut y_copy = y;
    loop {
        match y_copy.pop_front() {
            Option::Some(yi) => {
                let error = *yi - prediction;
                sum_squared_error += error.pow(FP16x16 { mag: 131072, sign: false } // 2
                );
            },
            Option::None(_) => {
                break;
            }
        };
    };

    sum_squared_error / FixedTrait::new_unscaled(y.len(), false)
}

fn best_split(data: Span<Span<FP16x16>>, target: Span<FP16x16>) -> (usize, FP16x16, FP16x16) {
    let mut best_mse = FP16x16 { mag: MAX, sign: false };
    let mut best_split_feature = 0;
    let mut best_split_value = FP16x16 { mag: 0, sign: false };
    let mut best_prediction = FP16x16 { mag: 0, sign: false };

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
                        let mut left_sum = FP16x16 { mag: 0, sign: false };
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
                        let left_target_as_fp: FP16x16 = FixedTrait::new_unscaled(
                            left_target.len(), false
                        );
                        let left_pred = left_sum / left_target_as_fp;

                        let mut right_sum = FP16x16 { mag: 0, sign: false };
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
                        let right_target_as_fp: FP16x16 = FixedTrait::new_unscaled(
                            right_target.len(), false
                        );
                        let right_pred = right_sum / right_target_as_fp;

                        let current_mse = (left_target_as_fp * mse(left_target.span(), left_pred))
                            + (right_target_as_fp * mse(right_target.span(), right_pred));

                        if current_mse < best_mse {
                            best_mse = current_mse;
                            best_split_feature = feature;
                            best_split_value = *value;

                            let mut total_sum = FP16x16 { mag: 0, sign: false };
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

                            best_prediction = total_sum
                                / FixedTrait::new_unscaled(target.len(), false);
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

    (best_split_feature, best_split_value, best_prediction)
}

fn build_tree(
    data: Span<Span<FP16x16>>, target: Span<FP16x16>, depth: usize, max_depth: usize
) -> TreeNode {
    if depth == max_depth || data.len() < 2 {
        let mut total = FP16x16 { mag: 0, sign: false };
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
        return TreeNode {
            left: Option::None(()),
            right: Option::None(()),
            split_feature: 0,
            split_value: FP16x16 { mag: 0, sign: false },
            prediction: total / FixedTrait::new_unscaled(target.len(), false),
        };
    }

    let (split_feature, split_value, prediction) = best_split(data, target);
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

    TreeNode {
        left: Option::Some(
            BoxTrait::new(build_tree(left_data.span(), left_target.span(), depth + 1, max_depth))
        ),
        right: Option::Some(
            BoxTrait::new(build_tree(right_data.span(), right_target.span(), depth + 1, max_depth))
        ),
        split_feature,
        split_value,
        prediction,
    }
}

#[test]
#[available_gas(2000000000000)]
fn test_tree() {
    let data = array![
        array![FixedTrait::new_unscaled(1, false), FixedTrait::new_unscaled(2, false)].span(),
        array![FixedTrait::new_unscaled(3, false), FixedTrait::new_unscaled(4, false)].span(),
        array![FixedTrait::new_unscaled(5, false), FixedTrait::new_unscaled(6, false)].span(),
        array![FixedTrait::new_unscaled(7, false), FixedTrait::<FP16x16>::new_unscaled(8, false)]
            .span(),
    ]
        .span();

    let target = array![
        FixedTrait::new_unscaled(2, false),
        FixedTrait::new_unscaled(4, false),
        FixedTrait::new_unscaled(6, false),
        FixedTrait::<FP16x16>::new_unscaled(8, false)
    ]
        .span();

    let mut tree = build_tree(data, target, 0, 3);

    let prediction_1 = tree
        .predict(
            array![FixedTrait::new_unscaled(1, false), FixedTrait::new_unscaled(2, false)].span()
        );

    let prediction_2 = tree
        .predict(
            array![FixedTrait::new_unscaled(3, false), FixedTrait::new_unscaled(4, false)].span()
        );

    let prediction_3 = tree
        .predict(
            array![FixedTrait::new_unscaled(5, false), FixedTrait::new_unscaled(6, false)].span()
        );

    let prediction_4 = tree
        .predict(
            array![FixedTrait::new_unscaled(7, false), FixedTrait::new_unscaled(8, false)].span()
        );

    assert(prediction_1 == FixedTrait::new_unscaled(2, false), 'should predict 2');
    assert(prediction_2 == FixedTrait::new_unscaled(4, false), 'should predict 4');
    assert(prediction_3 == FixedTrait::new_unscaled(6, false), 'should predict 6');
    assert(prediction_4 == FixedTrait::new_unscaled(8, false), 'should predict 8');
}
