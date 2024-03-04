# TreeEnsembleRegressor::predict

```rust 
   fn predict(regressor: TreeEnsembleRegressor<T>, X: Tensor<T>) -> (Span<usize>, MutMatrix::<T>);
```

Tree Ensemble regressor. Returns the regressed values for each input in N.

## Args

* `self`: TreeEnsembleRegressor<T> - A TreeEnsembleRegressor object.
* `X`:  Input 2D tensor.

## Returns

* Regressed values for each input in N

## Type Constraints

`TreeEnsembleRegressor` and `X` must be fixed points

## Examples

```rust
use orion::numbers::FP16x16;
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, U32Tensor};
use orion::operators::ml::{NODE_MODES, TreeEnsembleAttributes, TreeEnsemble};
use orion::operators::ml::tree_ensemble::tree_ensemble_regressor::{
    TreeEnsembleRegressor, POST_TRANSFORM, TreeEnsembleRegressorTrait, AGGREGATE_FUNCTION
};
use orion::operators::matrix::{MutMatrix, MutMatrixImpl};


fn tree_ensemble_regressor_helper(
    agg: AGGREGATE_FUNCTION
) -> (TreeEnsembleRegressor<FP16x16>, Tensor<FP16x16>) {
    let n_targets: usize = 1;
    let aggregate_function = agg;
    let nodes_falsenodeids: Span<usize> = array![4, 3, 0, 0, 0, 2, 0, 4, 0, 0].span();
    let nodes_featureids: Span<usize> = array![0, 2, 0, 0, 0, 0, 0, 2, 0, 0].span();
    let nodes_missing_value_tracks_true: Span<usize> = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0].span();
    let nodes_modes: Span<NODE_MODES> = array![
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF
    ]
        .span();
    let nodes_nodeids: Span<usize> = array![0, 1, 2, 3, 4, 0, 1, 2, 3, 4].span();
    let nodes_treeids: Span<usize> = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1].span();
    let nodes_truenodeids: Span<usize> = array![1, 2, 0, 0, 0, 1, 0, 3, 0, 0].span();
    let nodes_values: Span<FP16x16> = array![
        FP16x16 { mag: 17462, sign: false },
        FP16x16 { mag: 40726, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 47240, sign: true },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 36652, sign: true },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false }
    ]
        .span();
    let target_ids: Span<usize> = array![0, 0, 0, 0, 0, 0].span();
    let target_nodeids: Span<usize> = array![2, 3, 4, 1, 3, 4].span();
    let target_treeids: Span<usize> = array![0, 0, 0, 1, 1, 1].span();
    let target_weights: Span<FP16x16> = array![
        FP16x16 { mag: 5041, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 18724, sign: false },
        FP16x16 { mag: 32768, sign: false }
    ]
        .span();

    let base_values: Option<Span<FP16x16>> = Option::None;
    let post_transform = POST_TRANSFORM::NONE;

    let tree_ids: Span<usize> = array![0, 1].span();

    let mut root_index: Felt252Dict<usize> = Default::default();
    root_index.insert(0, 0);
    root_index.insert(1, 5);

    let mut node_index: Felt252Dict<usize> = Default::default();
    node_index
        .insert(2089986280348253421170679821480865132823066470938446095505822317253594081284, 0);
    node_index
        .insert(2001140082530619239661729809084578298299223810202097622761632384561112390979, 1);
    node_index
        .insert(2592670241084192212354027440049085852792506518781954896144296316131790403900, 2);
    node_index
        .insert(2960591271376829378356567803618548672034867345123727178628869426548453833420, 3);
    node_index
        .insert(458933264452572171106695256465341160654132084710250671055261382009315664425, 4);
    node_index
        .insert(1089549915800264549621536909767699778745926517555586332772759280702396009108, 5);
    node_index
        .insert(1321142004022994845681377299801403567378503530250467610343381590909832171180, 6);
    node_index
        .insert(2592987851775965742543459319508348457290966253241455514226127639100457844774, 7);
    node_index
        .insert(2492755623019086109032247218615964389726368532160653497039005814484393419348, 8);
    node_index
        .insert(1323616023845704258113538348000047149470450086307731200728039607710316625916, 9);

    let atts = TreeEnsembleAttributes {
        nodes_falsenodeids,
        nodes_featureids,
        nodes_missing_value_tracks_true,
        nodes_modes,
        nodes_nodeids,
        nodes_treeids,
        nodes_truenodeids,
        nodes_values
    };

    let mut ensemble: TreeEnsemble<FP16x16> = TreeEnsemble {
        atts, tree_ids, root_index, node_index
    };

    let mut regressor: TreeEnsembleRegressor<FP16x16> = TreeEnsembleRegressor {
        ensemble,
        target_ids,
        target_nodeids,
        target_treeids,
        target_weights,
        base_values,
        n_targets,
        aggregate_function,
        post_transform
    };

    let mut X: Tensor<FP16x16> = TensorTrait::new(
        array![3, 3].span(),
        array![
            FP16x16 { mag: 32768, sign: true },
            FP16x16 { mag: 26214, sign: true },
            FP16x16 { mag: 19660, sign: true },
            FP16x16 { mag: 13107, sign: true },
            FP16x16 { mag: 6553, sign: true },
            FP16x16 { mag: 0, sign: false },
            FP16x16 { mag: 6553, sign: false },
            FP16x16 { mag: 13107, sign: false },
            FP16x16 { mag: 19660, sign: false },
        ]
            .span()
    );

    (regressor, X)
}

fn test_tree_ensemble_regressor_SUM() -> MutMatrix::<FP16x16> {
    let (mut regressor, X) = tree_ensemble_regressor_helper(AGGREGATE_FUNCTION::SUM);
    let mut res = TreeEnsembleRegressorTrait::predict(regressor, X);
    res
}
>>> 

   [0.5769, 0.5769, 0.5769] 

```
