# TreeEnsemble::predict

```rust 
   fn predict(X: @Tensor<T>,
                  nodes_splits: Tensor<T>,
                  nodes_featureids: Span<usize>,
                  nodes_modes: Span<MODE>,
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
```

Tree Ensemble operator. Returns the regressed values for each input in a batch. Inputs have dimensions [N, F] where N is the input batch size and F is the number of input features. Outputs have dimensions [N, num_targets] where N is the batch size and num_targets is the number of targets, which is a configurable attribute.

## Args

* `X`:  Input 2D tensor.
* `nodes_splits`: Thresholds to do the splitting on for each node with mode that is not 'BRANCH_MEMBER'.
* `nodes_featureids`: Feature id for each node.
* `nodes_modes`: The comparison operation performed by the node. This is encoded as an enumeration of 'NODE_MODE::LEQ', 'NODE_MODE::LT', 'NODE_MODE::GTE', 'NODE_MODE::GT', 'NODE_MODE::EQ', 'NODE_MODE::NEQ', and 'NODE_MODE::MEMBER'
* `nodes_truenodeids`: If `nodes_trueleafs` is 0 (false) at an entry, this represents the position of the true branch node. 
* `nodes_falsenodeids`: If `nodes_falseleafs` is 0 (false) at an entry, this represents the position of the false branch node.
* `nodes_trueleafs`: 1 if true branch is leaf for each node and 0 an interior node.
* `nodes_falseleafs`: 1 if true branch is leaf for each node and 0 an interior node.
* `leaf_targetids`: The index of the target that this leaf contributes to (this must be in range `[0, n_targets)`).
* `leaf_weights`: The weight for each leaf.
* `tree_roots`: Index into `nodes_*` for the root of each tree. The tree structure is derived from the branching of each node.
* `post_transform`: Indicates the transform to apply to the score.One of 'POST_TRANSFORM::NONE', 'POST_TRANSFORM::SOFTMAX', 'POST_TRANSFORM::LOGISTIC', 'POST_TRANSFORM::SOFTMAX_ZERO'  or 'POST_TRANSFORM::PROBIT' ,
* `aggregate_function`: Defines how to aggregate leaf values within a target. One of 'AGGREGATE_FUNCTION::AVERAGE', 'AGGREGATE_FUNCTION::SUM', 'AGGREGATE_FUNCTION::MIN', 'AGGREGATE_FUNCTION::MAX` defaults to 'AGGREGATE_FUNCTION::SUM' 
* `nodes_hitrates`: Popularity of each node, used for performance and may be omitted.
* `nodes_missing_value_tracks_true`: For each node, define whether to follow the true branch (if attribute value is 1) or false branch (if attribute value is 0) in the presence of a NaN input feature. This attribute may be left undefined and the default value is false (0) for all nodes.
* `membership_values`: Members to test membership of for each set membership node. List all of the members to test again in the order that the 'BRANCH_MEMBER' mode appears in `node_modes`, delimited by `NaN`s. Will have the same number of sets of values as nodes with mode 'BRANCH_MEMBER'. This may be omitted if the node doesn't contain any 'BRANCH_MEMBER' nodes.
* `n_targets`: The total number of targets.


## Returns

* Output of shape [Batch Size, Number of targets]

## Type Constraints

`TreeEnsembleClassifier` and `X` must be fixed points

## Examples

```rust
use orion::numbers::FP16x16;
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, U32Tensor};
use orion::operators::ml::{TreeEnsembleTrait,POST_TRANSFORM, AGGREGATE_FUNCTION, NODE_MODE};
use orion::operators::matrix::{MutMatrix, MutMatrixImpl};
use orion::numbers::NumberTrait;

fn example_tree_ensemble_one_tree() ->  MutMatrix::<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 78643, sign: false });
    data.append(FP16x16 { mag: 222822, sign: false });
    data.append(FP16x16 { mag: 7864, sign: true });
    data.append(FP16x16 { mag: 108789, sign: false });
    data.append(FP16x16 { mag: 271319, sign: false });
    data.append(FP16x16 { mag: 115998, sign: false });
    let mut X = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 342753, sign: false });
    data.append(FP16x16 { mag: 794296, sign: false });
    data.append(FP16x16 { mag: 801505, sign: true });
    data.append(FP16x16 { mag: 472514, sign: false });
    let leaf_weights = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 205783, sign: false });
    data.append(FP16x16 { mag: 78643, sign: false });
    data.append(FP16x16 { mag: 275251, sign: false });
    let nodes_splits = TensorTrait::new(shape.span(), data.span());

    let membership_values = Option::None;

    let n_targets = 2;
    let aggregate_function = AGGREGATE_FUNCTION::SUM;
    let nodes_missing_value_tracks_true = Option::None;
    let nodes_hitrates = Option::None;
    let post_transform = POST_TRANSFORM::NONE;

    let tree_roots: Span<usize> = array![0].span();
    let nodes_modes: Span<MODE> = array![MODE::LEQ, MODE::LEQ, MODE::LEQ].span();

    let nodes_featureids: Span<usize> = array![0, 0, 0].span();
    let nodes_truenodeids: Span<usize> = array![1, 0, 1].span();
    let nodes_trueleafs: Span<usize> = array![0, 1, 1].span();
    let nodes_falsenodeids: Span<usize> = array![2, 2, 3].span();
    let nodes_falseleafs: Span<usize> = array![0, 1, 1].span();
    let leaf_targetids: Span<usize> = array![0, 1, 0, 1].span();

    return TreeEnsembleTrait::predict(
        @X,
        nodes_splits,
        nodes_featureids,
        nodes_modes,
        nodes_truenodeids,
        nodes_falsenodeids,
        nodes_trueleafs,
        nodes_falseleafs,
        leaf_targetids,
        leaf_weights,
        tree_roots,
        post_transform,
        aggregate_function,
        nodes_hitrates,
        nodes_missing_value_tracks_true,
        membership_values,
        n_targets
    );
}

>>> [[ 5.23  0.  ]
     [ 5.23  0.  ]
     [ 0.   12.12]]    
```
