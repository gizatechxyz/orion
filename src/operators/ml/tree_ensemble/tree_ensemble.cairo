use orion::operators::tensor::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;

use orion::operators::matrix::{MutMatrix, MutMatrixImpl, MutMatrixTrait};

#[derive(Copy, Drop)]
enum AGGREGATE_FUNCTION {
    AVERAGE,
    SUM,
    MIN,
    MAX,
}

#[derive(Copy, Drop)]
enum POST_TRANSFORM {
    NONE,
    SOFTMAX,
    LOGISTIC,
    SOFTMAX_ZERO,
    PROBIT,
}

#[derive(Copy, Drop)]
enum NODE_MODE {
    LEQ,
    LT,
    GTE,
    GT,
    EQ,
    NEQ,
    MEMBER,
}

/// Trait
///
/// predict - Returns the regressed values for each input in a batch.
trait TreeEnsembleTrait<T> {
    /// # TreeEnsemble::predict
    ///
    /// ```rust 
    ///    fn predict(X: @Tensor<T>,
    ///               nodes_splits: Tensor<T>,
    ///               nodes_featureids: Span<usize>,
    ///               nodes_modes: Span<MODE>,
    ///               nodes_truenodeids: Span<usize>,
    ///               nodes_falsenodeids: Span<usize>,
    ///               nodes_trueleafs: Span<usize>,
    ///               nodes_falseleafs: Span<usize>,
    ///               leaf_targetids: Span<usize>,
    ///               leaf_weights: Tensor<T>,
    ///               tree_roots: Span<usize>,
    ///               post_transform: POST_TRANSFORM,
    ///               aggregate_function: AGGREGATE_FUNCTION,
    ///               nodes_hitrates: Option<Tensor<T>>,
    ///               nodes_missing_value_tracks_true: Option<Span<usize>>,
    ///               membership_values: Option<Tensor<T>>,
    ///               n_targets: usize
    ///              ) -> MutMatrix::<T>;
    /// ```
    ///
    /// Tree Ensemble operator. Returns the regressed values for each input in a batch. Inputs have dimensions [N, F] where N is the input batch size and F is the number of input features. Outputs have dimensions [N, num_targets] where N is the batch size and num_targets is the number of targets, which is a configurable attribute.
    /// 
    /// ## Args
    ///
    /// * `X`:  Input 2D tensor.
    /// * `nodes_splits`: Thresholds to do the splitting on for each node with mode that is not 'BRANCH_MEMBER'.
    /// * `nodes_featureids`: Feature id for each node.
    /// * `nodes_modes`: The comparison operation performed by the node. This is encoded as an enumeration of 'NODE_MODE::LEQ', 'NODE_MODE::LT', 'NODE_MODE::GTE', 'NODE_MODE::GT', 'NODE_MODE::EQ', 'NODE_MODE::NEQ', and 'NODE_MODE::MEMBER'
    /// * `nodes_truenodeids`: If `nodes_trueleafs` is 0 (false) at an entry, this represents the position of the true branch node. 
    /// * `nodes_falsenodeids`: If `nodes_falseleafs` is 0 (false) at an entry, this represents the position of the false branch node.
    /// * `nodes_trueleafs`: 1 if true branch is leaf for each node and 0 an interior node.
    /// * `nodes_falseleafs`: 1 if true branch is leaf for each node and 0 an interior node.
    /// * `leaf_targetids`: The index of the target that this leaf contributes to (this must be in range `[0, n_targets)`).
    /// * `leaf_weights`: The weight for each leaf.
    /// * `tree_roots`: Index into `nodes_*` for the root of each tree. The tree structure is derived from the branching of each node.
    /// * `post_transform`: Indicates the transform to apply to the score.One of 'POST_TRANSFORM::NONE', 'POST_TRANSFORM::SOFTMAX', 'POST_TRANSFORM::LOGISTIC', 'POST_TRANSFORM::SOFTMAX_ZERO'  or 'POST_TRANSFORM::PROBIT' ,
    /// * `aggregate_function`: Defines how to aggregate leaf values within a target. One of 'AGGREGATE_FUNCTION::AVERAGE', 'AGGREGATE_FUNCTION::SUM', 'AGGREGATE_FUNCTION::MIN', 'AGGREGATE_FUNCTION::MAX` defaults to 'AGGREGATE_FUNCTION::SUM' 
    /// * `nodes_hitrates`: Popularity of each node, used for performance and may be omitted.
    /// * `nodes_missing_value_tracks_true`: For each node, define whether to follow the true branch (if attribute value is 1) or false branch (if attribute value is 0) in the presence of a NaN input feature. This attribute may be left undefined and the default value is false (0) for all nodes.
    /// * `membership_values`: Members to test membership of for each set membership node. List all of the members to test again in the order that the 'BRANCH_MEMBER' mode appears in `node_modes`, delimited by `NaN`s. Will have the same number of sets of values as nodes with mode 'BRANCH_MEMBER'. This may be omitted if the node doesn't contain any 'BRANCH_MEMBER' nodes.
    /// * `n_targets`: The total number of targets.

    ///
    /// ## Returns
    ///
    /// * Output of shape [Batch Size, Number of targets]
    ///
    /// ## Type Constraints
    ///
    /// `T` must be fixed point
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::FP16x16;
    /// use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, U32Tensor};
    /// use orion::operators::ml::{TreeEnsembleTrait,POST_TRANSFORM, AGGREGATE_FUNCTION, NODE_MODE};
    /// use orion::operators::matrix::{MutMatrix, MutMatrixImpl};
    /// use orion::numbers::NumberTrait;
    /// 
    /// fn example_tree_ensemble_one_tree() ->  MutMatrix::<FP16x16> {
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(3);
    ///     shape.append(2);
    /// 
    ///     let mut data = ArrayTrait::new();
    ///     data.append(FP16x16 { mag: 78643, sign: false });
    ///     data.append(FP16x16 { mag: 222822, sign: false });
    ///     data.append(FP16x16 { mag: 7864, sign: true });
    ///     data.append(FP16x16 { mag: 108789, sign: false });
    ///     data.append(FP16x16 { mag: 271319, sign: false });
    ///     data.append(FP16x16 { mag: 115998, sign: false });
    ///     let mut X = TensorTrait::new(shape.span(), data.span());
    /// 
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(4);
    /// 
    ///     let mut data = ArrayTrait::new();
    ///     data.append(FP16x16 { mag: 342753, sign: false });
    ///     data.append(FP16x16 { mag: 794296, sign: false });
    ///     data.append(FP16x16 { mag: 801505, sign: true });
    ///     data.append(FP16x16 { mag: 472514, sign: false });
    ///     let leaf_weights = TensorTrait::new(shape.span(), data.span());
    /// 
    ///     let mut shape = ArrayTrait::<usize>::new();
    ///     shape.append(3);
    /// 
    ///     let mut data = ArrayTrait::new();
    ///     data.append(FP16x16 { mag: 205783, sign: false });
    ///     data.append(FP16x16 { mag: 78643, sign: false });
    ///     data.append(FP16x16 { mag: 275251, sign: false });
    ///     let nodes_splits = TensorTrait::new(shape.span(), data.span());
    /// 
    ///     let membership_values = Option::None;
    /// 
    ///     let n_targets = 2;
    ///     let aggregate_function = AGGREGATE_FUNCTION::SUM;
    ///     let nodes_missing_value_tracks_true = Option::None;
    ///     let nodes_hitrates = Option::None;
    ///     let post_transform = POST_TRANSFORM::NONE;
    /// 
    ///     let tree_roots: Span<usize> = array![0].span();
    ///     let nodes_modes: Span<MODE> = array![MODE::LEQ, MODE::LEQ, MODE::LEQ].span();
    /// 
    ///     let nodes_featureids: Span<usize> = array![0, 0, 0].span();
    ///     let nodes_truenodeids: Span<usize> = array![1, 0, 1].span();
    ///     let nodes_trueleafs: Span<usize> = array![0, 1, 1].span();
    ///     let nodes_falsenodeids: Span<usize> = array![2, 2, 3].span();
    ///     let nodes_falseleafs: Span<usize> = array![0, 1, 1].span();
    ///     let leaf_targetids: Span<usize> = array![0, 1, 0, 1].span();
    /// 
    ///     return TreeEnsembleTrait::predict(
    ///         @X,
    ///         nodes_splits,
    ///         nodes_featureids,
    ///         nodes_modes,
    ///         nodes_truenodeids,
    ///         nodes_falsenodeids,
    ///         nodes_trueleafs,
    ///         nodes_falseleafs,
    ///         leaf_targetids,
    ///         leaf_weights,
    ///         tree_roots,
    ///         post_transform,
    ///         aggregate_function,
    ///         nodes_hitrates,
    ///         nodes_missing_value_tracks_true,
    ///         membership_values,
    ///         n_targets
    ///     );
    /// }
    ///
    /// >>> [[ 5.23  0.  ]
    ///      [ 5.23  0.  ]
    ///      [ 0.   12.12]]    
    /// ```
    ///
    fn predict(
        X: @Tensor<T>,
        nodes_splits: Tensor<T>,
        nodes_featureids: Span<usize>,
        nodes_modes: Span<NODE_MODE>,
        nodes_truenodeids: Span<usize>,
        nodes_falsenodeids: Span<usize>,
        nodes_trueleafs: Span<usize>,
        nodes_falseleafs: Span<usize>,
        leaf_targetids: Span<usize>,
        leaf_weights: Tensor<T>,
        tree_roots: Span<usize>,
        post_transform: POST_TRANSFORM,
        aggregate_function: AGGREGATE_FUNCTION,
        nodes_hitrates: Option<Tensor<T>>,
        nodes_missing_value_tracks_true: Option<Span<usize>>,
        membership_values: Option<Tensor<T>>,
        n_targets: usize
    ) -> MutMatrix::<T>;
}


impl TreeEnsembleImpl<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Copy<T>,
    +Drop<T>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +Add<T>,
    +Div<T>,
    +Mul<T>,
    +Into<usize, MAG>,
    +AddEq<T>,
> of TreeEnsembleTrait<T> {
    fn predict(
        X: @Tensor<T>,
        nodes_splits: Tensor<T>,
        nodes_featureids: Span<usize>,
        nodes_modes: Span<NODE_MODE>,
        nodes_truenodeids: Span<usize>,
        nodes_falsenodeids: Span<usize>,
        nodes_trueleafs: Span<usize>,
        nodes_falseleafs: Span<usize>,
        leaf_targetids: Span<usize>,
        leaf_weights: Tensor<T>,
        tree_roots: Span<usize>,
        post_transform: POST_TRANSFORM,
        aggregate_function: AGGREGATE_FUNCTION,
        nodes_hitrates: Option<Tensor<T>>,
        nodes_missing_value_tracks_true: Option<Span<usize>>,
        membership_values: Option<Tensor<T>>,
        n_targets: usize
    ) -> MutMatrix::<T> {
        let batch_size = *(*X).shape.at(0);
        let n_features = *(*X).shape.at(1);
        let n_trees = tree_roots.len();

        let mut set_membership_iter = array![].span();
        let mut map_member_to_nodeid = Default::default();

        let mut res: MutMatrix<T> = MutMatrixImpl::new(batch_size, n_targets);

        let (nodes_missing_value_tracks_true, nodes_missing_value_tracks_true_flag) =
            match nodes_missing_value_tracks_true {
            Option::Some(nodes_missing_value_tracks_true) => {
                (nodes_missing_value_tracks_true, true)
            },
            Option::None => { (array![].span(), false) }
        };

        match membership_values {
            Option::Some(membership_values) => {
                set_membership_iter = membership_values.data.clone();

                let mut tree_roots_iter = tree_roots.clone();
                loop {
                    match tree_roots_iter.pop_front() {
                        Option::Some(root_index) => {
                            let root_index = *root_index;
                            let is_leaf = (*nodes_trueleafs.at(root_index) == 1
                                && *nodes_falseleafs.at(root_index) == 1
                                && *nodes_truenodeids
                                    .at(root_index) == *nodes_falsenodeids
                                    .at(root_index));
                            map_members_to_nodeids(
                                root_index,
                                is_leaf,
                                nodes_modes,
                                nodes_truenodeids,
                                nodes_falsenodeids,
                                nodes_trueleafs,
                                nodes_falseleafs,
                                ref set_membership_iter,
                                ref map_member_to_nodeid,
                            );
                        },
                        Option::None => { break; }
                    }
                };
            },
            Option::None => {}
        };

        match aggregate_function {
            AGGREGATE_FUNCTION::AVERAGE => { res.set(batch_size, n_targets, NumberTrait::zero()); },
            AGGREGATE_FUNCTION::SUM => { res.set(batch_size, n_targets, NumberTrait::zero()); },
            AGGREGATE_FUNCTION::MIN => {
                let mut i = 0;
                while i != batch_size {
                    let mut j = 0;
                    while j != n_targets {
                        res.set(i, j, NumberTrait::min_value());
                        j += 1;
                    };
                    i += 1;
                };
            },
            AGGREGATE_FUNCTION::MAX => {
                let mut i = 0;
                while i != batch_size {
                    let mut j = 0;
                    while j != n_targets {
                        res.set(i, j, NumberTrait::max_value());
                        j += 1;
                    };
                    i += 1;
                };
            },
        }

        let mut weights = ArrayTrait::new();
        let mut target_ids = ArrayTrait::new();

        let mut tree_roots_iter = tree_roots.clone();
        loop {
            match tree_roots_iter.pop_front() {
                Option::Some(root_index) => {
                    let root_index = *root_index;
                    let is_leaf = (*nodes_trueleafs.at(root_index) == 1
                        && *nodes_falseleafs.at(root_index) == 1
                        && *nodes_truenodeids.at(root_index) == *nodes_falsenodeids.at(root_index));

                    let mut batch_num = 0;
                    while batch_num != batch_size {
                        let x_batch = SpanTrait::slice(
                            (*X).data, batch_num * n_features, n_features
                        );

                        let (weight, target) = iterate_node(
                            x_batch,
                            root_index,
                            is_leaf,
                            nodes_splits.data,
                            nodes_featureids,
                            nodes_modes,
                            nodes_truenodeids,
                            nodes_falsenodeids,
                            nodes_trueleafs,
                            nodes_falseleafs,
                            leaf_targetids,
                            leaf_weights,
                            nodes_hitrates,
                            nodes_missing_value_tracks_true,
                            nodes_missing_value_tracks_true_flag,
                            ref map_member_to_nodeid,
                        );
                        weights.append(weight);
                        target_ids.append(target);
                        batch_num += 1;
                    };
                },
                Option::None => { break; }
            }
        };

        let weights = weights.span();
        let target_ids = target_ids.span();

        let mut batch_num = 0;
        while batch_num != batch_size {
            match aggregate_function {
                AGGREGATE_FUNCTION::AVERAGE => {
                    let mut i = 0;
                    while i != n_trees {
                        let index = i * batch_size + batch_num;
                        res
                            .set(
                                batch_num,
                                *target_ids.at(index),
                                res.at(batch_num, *target_ids.at(index))
                                    + *weights.at(index)
                                        / NumberTrait::new_unscaled(n_trees.into(), false)
                            );
                        i += 1;
                    };
                },
                AGGREGATE_FUNCTION::SUM => {
                    let mut i = 0;
                    while i != n_trees {
                        let index = i * batch_size + batch_num;
                        res
                            .set(
                                batch_num,
                                *target_ids.at(index),
                                res.at(batch_num, *target_ids.at(index)) + *weights.at(index)
                            );
                        i += 1;
                    };
                },
                AGGREGATE_FUNCTION::MIN => {
                    let mut i = 0;
                    while i != n_targets {
                        let val = NumberTrait::min(
                            res.at(batch_num, *target_ids.at(batch_num)), *weights.at(batch_num)
                        );
                        res.set(batch_num, *target_ids.at(batch_num), val);
                        i += 1;
                    };
                },
                AGGREGATE_FUNCTION::MAX => {
                    let mut i = 0;
                    while i != n_targets {
                        let val = NumberTrait::max(
                            res.at(batch_num, *target_ids.at(batch_num)), *weights.at(batch_num)
                        );
                        res.set(batch_num, *target_ids.at(batch_num), val);
                        i += 1;
                    };
                }
            }

            batch_num += 1;
        };

        // Post Transform
        let mut res = match post_transform {
            POST_TRANSFORM::NONE => res,
            POST_TRANSFORM::SOFTMAX => res.softmax(1),
            POST_TRANSFORM::LOGISTIC => res.sigmoid(),
            POST_TRANSFORM::SOFTMAX_ZERO => res.softmax_zero(1),
            POST_TRANSFORM::PROBIT => core::panic_with_felt252('Probit not supported yet'),
        };

        return res;
    }
}
fn iterate_node<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Copy<T>,
    +Drop<T>,
    +PartialOrd<T>,
    +PartialEq<T>,
>(
    X: Span<T>,
    current_node_index: usize,
    is_leaf: bool,
    nodes_splits: Span<T>,
    nodes_featureids: Span<usize>,
    nodes_modes: Span<NODE_MODE>,
    nodes_truenodeids: Span<usize>,
    nodes_falsenodeids: Span<usize>,
    nodes_trueleafs: Span<usize>,
    nodes_falseleafs: Span<usize>,
    leaf_targetids: Span<usize>,
    leaf_weights: Tensor<T>,
    nodes_hitrates: Option<Tensor<T>>,
    nodes_missing_value_tracks_true: Span<usize>,
    nodes_missing_value_tracks_true_flag: bool,
    ref map_member_to_nodeid: Felt252Dict<Nullable<Span<T>>>,
) -> (T, usize) {
    let mut current_node_index = current_node_index;
    let mut is_leaf = is_leaf;

    while !is_leaf {
        let nmvtt_flag = if nodes_missing_value_tracks_true_flag {
            *nodes_missing_value_tracks_true.at(current_node_index) == 1
        } else {
            nodes_missing_value_tracks_true_flag
        };
        if compare(
            *X.at(*nodes_featureids.at(current_node_index)),
            current_node_index,
            *nodes_splits.at(current_node_index),
            *nodes_modes.at(current_node_index),
            ref map_member_to_nodeid,
            nmvtt_flag
        ) {
            is_leaf = *nodes_trueleafs.at(current_node_index) == 1;
            current_node_index = *nodes_truenodeids.at(current_node_index);
        } else {
            is_leaf = *nodes_falseleafs.at(current_node_index) == 1;
            current_node_index = *nodes_falsenodeids.at(current_node_index);
        };
    };

    return (*leaf_weights.data.at(current_node_index), *leaf_targetids.at(current_node_index));
}

fn map_members_to_nodeids<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Copy<T>,
    +Drop<T>,
    +PartialOrd<T>,
    +PartialEq<T>,
>(
    current_node_index: usize,
    is_leaf: bool,
    nodes_modes: Span<NODE_MODE>,
    nodes_truenodeids: Span<usize>,
    nodes_falsenodeids: Span<usize>,
    nodes_trueleafs: Span<usize>,
    nodes_falseleafs: Span<usize>,
    ref set_membership_iter: Span<T>,
    ref map_member_to_nodeid: Felt252Dict<Nullable<Span<T>>>,
) {
    let mut current_node_index = current_node_index;
    let mut is_leaf = is_leaf;

    if is_leaf {
        return;
    }

    match *nodes_modes.at(current_node_index) {
        NODE_MODE::LEQ => {},
        NODE_MODE::LT => {},
        NODE_MODE::GTE => {},
        NODE_MODE::GT => {},
        NODE_MODE::EQ => {},
        NODE_MODE::NEQ => {},
        NODE_MODE::MEMBER => {
            let mut subset_members = ArrayTrait::new();
            loop {
                match set_membership_iter.pop_front() {
                    Option::Some(v) => {
                        if *v == NumberTrait::NaN() {
                            break;
                        }
                        subset_members.append(*v)
                    },
                    Option::None => { break; }
                }
            };
            map_member_to_nodeid
                .insert(current_node_index.into(), NullableTrait::new(subset_members.span()));
        },
    }
    // true branch
    map_members_to_nodeids(
        *nodes_truenodeids.at(current_node_index),
        *nodes_trueleafs.at(current_node_index) == 1,
        nodes_modes,
        nodes_truenodeids,
        nodes_falsenodeids,
        nodes_trueleafs,
        nodes_falseleafs,
        ref set_membership_iter,
        ref map_member_to_nodeid,
    );

    // false branch
    map_members_to_nodeids(
        *nodes_falsenodeids.at(current_node_index),
        *nodes_falseleafs.at(current_node_index) == 1,
        nodes_modes,
        nodes_truenodeids,
        nodes_falsenodeids,
        nodes_trueleafs,
        nodes_falseleafs,
        ref set_membership_iter,
        ref map_member_to_nodeid,
    );
}


fn compare<
    T, MAG, +TensorTrait<T>, +NumberTrait<T, MAG>, +Copy<T>, +Drop<T>, +PartialOrd<T>, +PartialEq<T>
>(
    x_feat: T,
    current_node_index: usize,
    value: T,
    mode: NODE_MODE,
    ref map_members_to_nodeids: Felt252Dict<Nullable<Span<T>>>,
    nodes_missing_value_tracks_true_flag: bool,
) -> bool {
    match mode {
        NODE_MODE::LEQ => {
            (x_feat <= value && !x_feat.is_nan()) || nodes_missing_value_tracks_true_flag
        },
        NODE_MODE::LT => {
            (x_feat < value && !x_feat.is_nan()) || nodes_missing_value_tracks_true_flag
        },
        NODE_MODE::GTE => {
            (x_feat >= value && !x_feat.is_nan()) || nodes_missing_value_tracks_true_flag
        },
        NODE_MODE::GT => {
            (x_feat > value && !x_feat.is_nan()) || nodes_missing_value_tracks_true_flag
        },
        NODE_MODE::EQ => {
            (x_feat == value && !x_feat.is_nan()) || nodes_missing_value_tracks_true_flag
        },
        NODE_MODE::NEQ => {
            (x_feat != value && !x_feat.is_nan()) || nodes_missing_value_tracks_true_flag
        },
        NODE_MODE::MEMBER => {
            let mut set_members = map_members_to_nodeids.get(current_node_index.into()).deref();
            loop {
                match set_members.pop_front() {
                    Option::Some(v) => { if x_feat == *v {
                        break true;
                    } },
                    Option::None => { break false; }
                }
            }
        },
    }
}
