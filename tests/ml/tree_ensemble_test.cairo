use orion::numbers::FP16x16;
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, U32Tensor};
use orion::operators::ml::tree_ensemble::tree_ensemble::{
    TreeEnsembleTrait, POST_TRANSFORM, AGGREGATE_FUNCTION, NODE_MODE
};
use orion::operators::matrix::{MutMatrix, MutMatrixImpl, MutMatrixTrait};
use orion::numbers::NumberTrait;


#[test]
#[available_gas(200000000000)]
fn export_tree_ensemble_two_trees() {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 32768, sign: true });
    data.append(FP16x16 { mag: 26214, sign: true });
    data.append(FP16x16 { mag: 19660, sign: true });
    data.append(FP16x16 { mag: 13107, sign: true });
    data.append(FP16x16 { mag: 6553, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 6553, sign: false });
    data.append(FP16x16 { mag: 13107, sign: false });
    data.append(FP16x16 { mag: 19660, sign: false });
    let mut X = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 5041, sign: false });
    data.append(FP16x16 { mag: 32768, sign: false });
    data.append(FP16x16 { mag: 32768, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 18724, sign: false });
    data.append(FP16x16 { mag: 32768, sign: false });
    let leaf_weights = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 17462, sign: false });
    data.append(FP16x16 { mag: 40726, sign: false });
    data.append(FP16x16 { mag: 36652, sign: true });
    data.append(FP16x16 { mag: 47240, sign: true });
    let nodes_splits = TensorTrait::new(shape.span(), data.span());

    let n_targets = 1;
    let aggregate_function = AGGREGATE_FUNCTION::AVERAGE;
    let nodes_missing_value_tracks_true = Option::None;
    let nodes_hitrates = Option::None;
    let post_transform = POST_TRANSFORM::NONE;

    let tree_roots: Span<usize> = array![0, 2].span();
    let nodes_modes: Span<NODE_MODE> = array![
        NODE_MODE::LEQ, NODE_MODE::LEQ, NODE_MODE::LEQ, NODE_MODE::LEQ
    ]
        .span();

    let nodes_featureids: Span<usize> = array![0, 2, 0, 0].span();
    let nodes_truenodeids: Span<usize> = array![1, 0, 3, 4].span();
    let nodes_trueleafs: Span<usize> = array![0, 1, 1, 1].span();
    let nodes_falsenodeids: Span<usize> = array![2, 1, 3, 5].span();
    let nodes_falseleafs: Span<usize> = array![1, 1, 0, 1].span();
    let leaf_targetids: Span<usize> = array![0, 0, 0, 0, 0, 0].span();

    let mut scores = TreeEnsembleTrait::predict(
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
        Option::None,
        n_targets
    );

    // ASSERT SCOREs
    assert(scores.at(0, 0) == FP16x16 { mag: 18904, sign: false }, 'scores.at(0, 0)');
    assert(scores.at(1, 0) == FP16x16 { mag: 18904, sign: false }, 'scores.at(1, 0)');
    assert(scores.at(2, 0) == FP16x16 { mag: 18904, sign: false }, 'scores.at(2, 0)');
}


#[test]
#[available_gas(200000000000)]
fn export_tree_ensemble_one_tree() {
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
    let nodes_modes: Span<NODE_MODE> = array![NODE_MODE::LEQ, NODE_MODE::LEQ, NODE_MODE::LEQ]
        .span();

    let nodes_featureids: Span<usize> = array![0, 0, 0].span();
    let nodes_truenodeids: Span<usize> = array![1, 0, 1].span();
    let nodes_trueleafs: Span<usize> = array![0, 1, 1].span();
    let nodes_falsenodeids: Span<usize> = array![2, 2, 3].span();
    let nodes_falseleafs: Span<usize> = array![0, 1, 1].span();
    let leaf_targetids: Span<usize> = array![0, 1, 0, 1].span();

    let mut scores = TreeEnsembleTrait::predict(
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

    // ASSERT SCOREs
    assert(scores.at(0, 0) == FP16x16 { mag: 342753, sign: false }, 'scores.at(0, 0)');
    assert(scores.at(0, 1) == FP16x16 { mag: 0, sign: false }, 'scores.at(0, 1)');

    assert(scores.at(1, 0) == FP16x16 { mag: 342753, sign: false }, 'scores.at(1, 0)');
    assert(scores.at(1, 1) == FP16x16 { mag: 0, sign: false }, 'scores.at(1, 1)');

    assert(scores.at(2, 0) == FP16x16 { mag: 0, sign: false }, 'scores.at(2, 0)');
    assert(scores.at(2, 1) == FP16x16 { mag: 794296, sign: false }, 'scores.at(2, 1)');
}


#[test]
#[available_gas(200000000000)]
fn export_tree_ensemble_set_membership() {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 78643, sign: false });
    data.append(FP16x16 { mag: 222822, sign: false });
    data.append(FP16x16 { mag: 7864, sign: true });
    data.append(NumberTrait::<FP16x16>::NaN());
    data.append(FP16x16 { mag: 786432, sign: false });
    data.append(FP16x16 { mag: 458752, sign: false });
    let mut X = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 65536000, sign: false });
    data.append(FP16x16 { mag: 6553600, sign: false });
    let leaf_weights = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 720896, sign: false });
    data.append(FP16x16 { mag: 1522663424, sign: false });
    data.append(NumberTrait::<FP16x16>::NaN());
    let nodes_splits = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(8);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 78643, sign: false });
    data.append(FP16x16 { mag: 242483, sign: false });
    data.append(FP16x16 { mag: 524288, sign: false });
    data.append(FP16x16 { mag: 589824, sign: false });
    data.append(NumberTrait::<FP16x16>::NaN());
    data.append(FP16x16 { mag: 786432, sign: false });
    data.append(FP16x16 { mag: 458752, sign: false });
    data.append(NumberTrait::<FP16x16>::NaN());
    let membership_values = Option::Some(TensorTrait::new(shape.span(), data.span()));

    let n_targets = 4;
    let aggregate_function = AGGREGATE_FUNCTION::SUM;
    let nodes_missing_value_tracks_true = Option::None;
    let nodes_hitrates = Option::None;
    let post_transform = POST_TRANSFORM::NONE;

    let tree_roots: Span<usize> = array![0].span();
    let nodes_modes: Span<NODE_MODE> = array![NODE_MODE::LEQ, NODE_MODE::MEMBER, NODE_MODE::MEMBER]
        .span();

    let nodes_featureids: Span<usize> = array![0, 0, 0].span();
    let nodes_truenodeids: Span<usize> = array![1, 0, 1].span();
    let nodes_trueleafs: Span<usize> = array![0, 1, 1].span();
    let nodes_falsenodeids: Span<usize> = array![2, 2, 3].span();
    let nodes_falseleafs: Span<usize> = array![1, 0, 1].span();
    let leaf_targetids: Span<usize> = array![0, 1, 2, 3].span();

    let mut scores = TreeEnsembleTrait::predict(
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

    // ASSERT SCOREs
    assert(scores.at(0, 0) == FP16x16 { mag: 65536, sign: false }, 'scores.at(0, 0)');
    assert(scores.at(0, 1) == FP16x16 { mag: 0, sign: false }, 'scores.at(0, 1)');
    assert(scores.at(0, 2) == FP16x16 { mag: 0, sign: false }, 'scores.at(0, 2)');
    assert(scores.at(0, 3) == FP16x16 { mag: 0, sign: false }, 'scores.at(0, 3)');

    assert(scores.at(1, 0) == FP16x16 { mag: 0, sign: false }, 'scores.at(1, 0)');
    assert(scores.at(1, 1) == FP16x16 { mag: 0, sign: false }, 'scores.at(1, 1)');
    assert(scores.at(1, 2) == FP16x16 { mag: 0, sign: false }, 'scores.at(1, 2)');
    assert(scores.at(1, 3) == FP16x16 { mag: 6553600, sign: false }, 'scores.at(1, 3)');

    assert(scores.at(2, 0) == FP16x16 { mag: 0, sign: false }, 'scores.at(2, 0)');
    assert(scores.at(2, 1) == FP16x16 { mag: 0, sign: false }, 'scores.at(2, 1)');
    assert(scores.at(2, 2) == FP16x16 { mag: 0, sign: false }, 'scores.at(2, 2)');
    assert(scores.at(2, 3) == FP16x16 { mag: 6553600, sign: false }, 'scores.at(2, 3)');

    assert(scores.at(3, 0) == FP16x16 { mag: 0, sign: false }, 'scores.at(3, 0)');
    assert(scores.at(3, 1) == FP16x16 { mag: 0, sign: false }, 'scores.at(3, 1)');
    assert(scores.at(3, 2) == FP16x16 { mag: 65536000, sign: false }, 'scores.at(3, 2)');
    assert(scores.at(3, 3) == FP16x16 { mag: 0, sign: false }, 'scores.at(3, 3)');

    assert(scores.at(4, 0) == FP16x16 { mag: 0, sign: false }, 'scores.at(4, 0)');
    assert(scores.at(4, 1) == FP16x16 { mag: 0, sign: false }, 'scores.at(4, 1)');
    assert(scores.at(4, 2) == FP16x16 { mag: 65536000, sign: false }, 'scores.at(4, 2)');
    assert(scores.at(4, 3) == FP16x16 { mag: 0, sign: false }, 'scores.at(4, 3)');

    assert(scores.at(5, 0) == FP16x16 { mag: 0, sign: false }, 'scores.at(5, 0)');
    assert(scores.at(5, 1) == FP16x16 { mag: 655360, sign: false }, 'scores.at(5, 1)');
    assert(scores.at(5, 2) == FP16x16 { mag: 0, sign: false }, 'scores.at(5, 2)');
    assert(scores.at(5, 3) == FP16x16 { mag: 0, sign: false }, 'scores.at(5, 3)');
}

