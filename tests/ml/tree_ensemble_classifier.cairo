use orion::numbers::FP16x16;
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, U32Tensor};
use orion::operators::ml::tree_ensemble::core::{NODE_MODES, TreeEnsembleAttributes, TreeEnsemble};
use orion::operators::ml::tree_ensemble::tree_ensemble_classifier::{
    TreeEnsembleClassifier, POST_TRANSFORM, TreeEnsembleClassifierTrait
};
use orion::operators::tensor::implementations::tensor_fp16x16::relative_eq;
use orion::operators::matrix::matrix::{MutMatrix, MutMatrixImpl};

#[test]
#[available_gas(200000000000)]
fn test_tree_ensemble_classifier_multi_pt_none() {
    let (mut classifier, X) = tree_ensemble_classifier_helper(POST_TRANSFORM::NONE);

    let (labels, mut scores) = TreeEnsembleClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 0, 'labels[0]');
    assert(*labels[1] == 0, 'labels[1]');
    assert(*labels[2] == 1, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    assert(
        relative_eq(@scores.get(0, 0).unwrap(), @FP16x16 { mag: 60075, sign: false }) == true,
        'score[0, 0]'
    );
    assert(
        relative_eq(@scores.get(0, 1).unwrap(), @FP16x16 { mag: 0, sign: false }) == true,
        'score[0, 1]'
    );
    assert(
        relative_eq(@scores.get(0, 2).unwrap(), @FP16x16 { mag: 5461, sign: false }) == true,
        'score[0, 2]'
    );
    assert(
        relative_eq(@scores.get(1, 0).unwrap(), @FP16x16 { mag: 37329, sign: false }) == true,
        'score[1, 0]'
    );
    assert(
        relative_eq(@scores.get(1, 1).unwrap(), @FP16x16 { mag: 12528, sign: false }) == true,
        'score[1, 1]'
    );
    assert(
        relative_eq(@scores.get(1, 2).unwrap(), @FP16x16 { mag: 15677, sign: false }) == true,
        'score[1, 2]'
    );
    assert(
        relative_eq(@scores.get(2, 0).unwrap(), @FP16x16 { mag: 19853, sign: false }) == true,
        'score[2, 0]'
    );
    assert(
        relative_eq(@scores.get(2, 1).unwrap(), @FP16x16 { mag: 28257, sign: false }) == true,
        'score[2, 1]'
    );
    assert(
        relative_eq(@scores.get(2, 2).unwrap(), @FP16x16 { mag: 17424, sign: false }) == true,
        'score[2, 2]'
    );
}

#[test]
#[available_gas(200000000000)]
fn test_tree_ensemble_classifier_multi_pt_softmax() {
    let (mut classifier, X) = tree_ensemble_classifier_helper(POST_TRANSFORM::SOFTMAX);

    let (labels, mut scores) = TreeEnsembleClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 0, 'labels[0]');
    assert(*labels[1] == 0, 'labels[1]');
    assert(*labels[2] == 1, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    assert(
        relative_eq(@scores.get(0, 0).unwrap(), @FP16x16 { mag: 35725, sign: false }) == true,
        'score[0, 0]'
    );
    assert(
        relative_eq(@scores.get(0, 1).unwrap(), @FP16x16 { mag: 14284, sign: false }) == true,
        'score[0, 1]'
    );
    assert(
        relative_eq(@scores.get(0, 2).unwrap(), @FP16x16 { mag: 15526, sign: false }) == true,
        'score[0, 2]'
    );
    assert(
        relative_eq(@scores.get(1, 0).unwrap(), @FP16x16 { mag: 27266, sign: false }) == true,
        'score[1, 0]'
    );
    assert(
        relative_eq(@scores.get(1, 1).unwrap(), @FP16x16 { mag: 18675, sign: false }) == true,
        'score[1, 1]'
    );
    assert(
        relative_eq(@scores.get(1, 2).unwrap(), @FP16x16 { mag: 19594, sign: false }) == true,
        'score[1, 2]'
    );
    assert(
        relative_eq(@scores.get(2, 0).unwrap(), @FP16x16 { mag: 21137, sign: false }) == true,
        'score[2, 0]'
    );
    assert(
        relative_eq(@scores.get(2, 1).unwrap(), @FP16x16 { mag: 24029, sign: false }) == true,
        'score[2, 1]'
    );
    assert(
        relative_eq(@scores.get(2, 2).unwrap(), @FP16x16 { mag: 20368, sign: false }) == true,
        'score[2, 2]'
    );
}

#[test]
#[available_gas(200000000000)]
fn test_tree_ensemble_classifier_multi_pt_softmax_zero() {
    let (mut classifier, X) = tree_ensemble_classifier_helper(POST_TRANSFORM::SOFTMAXZERO);

    let (labels, mut scores) = TreeEnsembleClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 0, 'labels[0] == 0');
    assert(*labels[1] == 0, 'labels[1] == 0');
    assert(*labels[2] == 1, 'labels[2] == 1');
    assert(labels.len() == 3, 'len(labels) == 3');

    // ASSERT SCORES
    assert(
        relative_eq(@scores.get(0, 0).unwrap(), @FP16x16 { mag: 45682, sign: false }) == true,
        'score[0, 0]'
    );
    assert(
        relative_eq(@scores.get(0, 1).unwrap(), @FP16x16 { mag: 0, sign: false }) == true,
        'score[0, 1]'
    );
    assert(
        relative_eq(@scores.get(0, 2).unwrap(), @FP16x16 { mag: 19853, sign: false }) == true,
        'score[0, 2]'
    );
    assert(
        relative_eq(@scores.get(1, 0).unwrap(), @FP16x16 { mag: 27266, sign: false }) == true,
        'score[1, 0]'
    );
    assert(
        relative_eq(@scores.get(1, 1).unwrap(), @FP16x16 { mag: 18675, sign: false }) == true,
        'score[1, 1]'
    );
    assert(
        relative_eq(@scores.get(1, 2).unwrap(), @FP16x16 { mag: 19594, sign: false }) == true,
        'score[1, 2]'
    );
    assert(
        relative_eq(@scores.get(2, 0).unwrap(), @FP16x16 { mag: 21137, sign: false }) == true,
        'score[2, 0]'
    );
    assert(
        relative_eq(@scores.get(2, 1).unwrap(), @FP16x16 { mag: 24029, sign: false }) == true,
        'score[2, 1]'
    );
    assert(
        relative_eq(@scores.get(2, 2).unwrap(), @FP16x16 { mag: 20368, sign: false }) == true,
        'score[2, 2]'
    );
}


#[test]
#[available_gas(200000000000)]
fn test_tree_ensemble_classifier_multi_pt_logistic() {
    let (mut classifier, X) = tree_ensemble_classifier_helper(POST_TRANSFORM::LOGISTIC);

    let (labels, mut scores) = TreeEnsembleClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 0, 'labels[0] == 0');
    assert(*labels[1] == 0, 'labels[1] == 0');
    assert(*labels[2] == 1, 'labels[2] == 1');
    assert(labels.len() == 3, 'len(labels) == 3');

    // ASSERT SCORES
    assert(
        relative_eq(@scores.get(0, 0).unwrap(), @FP16x16 { mag: 46816, sign: false }) == true,
        'score[0, 0]'
    );
    assert(
        relative_eq(@scores.get(0, 1).unwrap(), @FP16x16 { mag: 32768, sign: false }) == true,
        'score[0, 1]'
    );
    assert(
        relative_eq(@scores.get(0, 2).unwrap(), @FP16x16 { mag: 34132, sign: false }) == true,
        'score[0, 2]'
    );
    assert(
        relative_eq(@scores.get(1, 0).unwrap(), @FP16x16 { mag: 41856, sign: false }) == true,
        'score[1, 0]'
    );
    assert(
        relative_eq(@scores.get(1, 1).unwrap(), @FP16x16 { mag: 35890, sign: false }) == true,
        'score[1, 1]'
    );
    assert(
        relative_eq(@scores.get(1, 2).unwrap(), @FP16x16 { mag: 36668, sign: false }) == true,
        'score[1, 2]'
    );
    assert(
        relative_eq(@scores.get(2, 0).unwrap(), @FP16x16 { mag: 37693, sign: false }) == true,
        'score[2, 0]'
    );
    assert(
        relative_eq(@scores.get(2, 1).unwrap(), @FP16x16 { mag: 39724, sign: false }) == true,
        'score[2, 1]'
    );
    assert(
        relative_eq(@scores.get(2, 2).unwrap(), @FP16x16 { mag: 37098, sign: false }) == true,
        'score[2, 2]'
    );
}

#[test]
#[available_gas(200000000000)]
fn test_tree_ensemble_classifier_binary_none() {
    let (mut classifier, X) = tree_ensemble_classifier_binary_class_helper(POST_TRANSFORM::NONE);

    let (labels, mut scores) = TreeEnsembleClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 1, 'labels[0]');
    assert(labels.len() == 1, 'len(labels)');

    // ASSERT SCORES
    assert(
        relative_eq(@scores.get(0, 0).unwrap(), @FP16x16 { mag: 0, sign: false }) == true,
        'score[0, 0]'
    );
    assert(
        relative_eq(@scores.get(0, 1).unwrap(), @FP16x16 { mag: 65536, sign: false }) == true,
        'score[0, 1]'
    );
}

#[test]
#[available_gas(200000000000)]
fn test_tree_ensemble_classifier_binary_logistic() {
    let (mut classifier, X) = tree_ensemble_classifier_binary_class_helper(
        POST_TRANSFORM::LOGISTIC
    );

    let (labels, mut scores) = TreeEnsembleClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 1, 'labels[0]');
    assert(labels.len() == 1, 'len(labels)');

    // ASSERT SCORES
    assert(
        relative_eq(@scores.get(0, 0).unwrap(), @FP16x16 { mag: 17625, sign: false }) == true,
        'score[0, 0]'
    );
    assert(
        relative_eq(@scores.get(0, 1).unwrap(), @FP16x16 { mag: 47910, sign: false }) == true,
        'score[0, 1]'
    );
}

#[test]
#[available_gas(200000000000)]
fn test_tree_ensemble_classifier_binary_softmax() {
    let (mut classifier, X) = tree_ensemble_classifier_binary_class_helper(POST_TRANSFORM::SOFTMAX);

    let (labels, mut scores) = TreeEnsembleClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 1, 'labels[0]');
    assert(labels.len() == 1, 'len(labels)');

    // ASSERT SCORES
    assert(
        relative_eq(@scores.get(0, 0).unwrap(), @FP16x16 { mag: 7812, sign: false }) == true,
        'score[0, 0]'
    );
    assert(
        relative_eq(@scores.get(0, 1).unwrap(), @FP16x16 { mag: 57723, sign: false }) == true,
        'score[0, 1]'
    );
}

#[test]
#[available_gas(200000000000)]
fn test_tree_ensemble_classifier_binary_softmax_zero() {
    let (mut classifier, X) = tree_ensemble_classifier_binary_class_helper(
        POST_TRANSFORM::SOFTMAXZERO
    );

    let (labels, mut scores) = TreeEnsembleClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 1, 'labels[0]');
    assert(labels.len() == 1, 'len(labels)');

    // ASSERT SCORES
    assert(
        relative_eq(@scores.get(0, 0).unwrap(), @FP16x16 { mag: 7812, sign: false }) == true,
        'score[0, 0]'
    );
    assert(
        relative_eq(@scores.get(0, 1).unwrap(), @FP16x16 { mag: 57723, sign: false }) == true,
        'score[0, 1]'
    );
}

// #[test]
// #[available_gas(200000000000)]
// fn test_tree_ensemble_classifier_binary_probit() {
//     let (mut classifier, X) = tree_ensemble_classifier_binary_class_helper(POST_TRANSFORM::PROBIT);

//     let (labels, mut scores) = TreeEnsembleClassifierTrait::predict(ref classifier, X);

//     // ASSERT LABELS
//     assert(*labels[0] == 1, 'labels[0]');
//     assert(labels.len() == 1, 'len(labels)');

//     // ASSERT SCORES
//     assert(
//         relative_eq(@scores.get(0, 0).unwrap(), @FP16x16 { mag: 0, sign: false }) == true,
//         'score[0, 0]'
//     );
//     assert(
//         relative_eq(@scores.get(0, 1).unwrap(), @FP16x16 { mag: 65536, sign: false }) == true,
//         'score[0, 1]'
//     );
// }

// ============ HELPER ============ //

fn tree_ensemble_classifier_helper(
    post_transform: POST_TRANSFORM
) -> (TreeEnsembleClassifier<FP16x16>, Tensor<FP16x16>) {
    let class_ids: Span<usize> = array![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        .span();

    let class_nodeids: Span<usize> = array![2, 2, 2, 3, 3, 3, 4, 4, 4, 1, 1, 1, 3, 3, 3, 4, 4, 4]
        .span();

    let class_treeids: Span<usize> = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        .span();

    let class_weights: Span<FP16x16> = array![
        FP16x16 { mag: 30583, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 2185, sign: false },
        FP16x16 { mag: 13107, sign: false },
        FP16x16 { mag: 15729, sign: false },
        FP16x16 { mag: 3932, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 29491, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 3277, sign: false },
        FP16x16 { mag: 6746, sign: false },
        FP16x16 { mag: 12529, sign: false },
        FP16x16 { mag: 13493, sign: false },
    ]
        .span();

    let classlabels: Span<usize> = array![0, 1, 2].span();

    let nodes_falsenodeids: Span<usize> = array![4, 3, 0, 0, 0, 2, 0, 4, 0, 0].span();

    let nodes_featureids: Span<usize> = array![1, 0, 0, 0, 0, 1, 0, 0, 0, 0].span();

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
        NODE_MODES::LEAF,
    ]
        .span();

    let nodes_nodeids: Span<usize> = array![0, 1, 2, 3, 4, 0, 1, 2, 3, 4].span();

    let nodes_treeids: Span<usize> = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1].span();

    let nodes_truenodeids: Span<usize> = array![1, 2, 0, 0, 0, 1, 0, 3, 0, 0].span();

    let nodes_values: Span<FP16x16> = array![
        FP16x16 { mag: 81892, sign: false },
        FP16x16 { mag: 19992, sign: true },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 110300, sign: true },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 44245, sign: true },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
    ]
        .span();

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

    let base_values: Option<Span<FP16x16>> = Option::None;

    let mut classifier: TreeEnsembleClassifier<FP16x16> = TreeEnsembleClassifier {
        ensemble,
        class_ids,
        class_nodeids,
        class_treeids,
        class_weights,
        classlabels,
        base_values,
        post_transform
    };

    let mut X: Tensor<FP16x16> = TensorTrait::new(
        array![3, 3].span(),
        array![
            FP16x16 { mag: 65536, sign: true },
            FP16x16 { mag: 52429, sign: true },
            FP16x16 { mag: 39322, sign: true },
            FP16x16 { mag: 26214, sign: true },
            FP16x16 { mag: 13107, sign: true },
            FP16x16 { mag: 0, sign: false },
            FP16x16 { mag: 13107, sign: false },
            FP16x16 { mag: 26214, sign: false },
            FP16x16 { mag: 39322, sign: false },
        ]
            .span()
    );

    (classifier, X)
}

// ============ BINARY CLASS HELPER ============ //

fn tree_ensemble_classifier_binary_class_helper(
    post_transform: POST_TRANSFORM
) -> (TreeEnsembleClassifier<FP16x16>, Tensor<FP16x16>) {
    let class_ids: Span<usize> = array![
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ]
        .span();
    let class_nodeids: Span<usize> = array![
        4,
        5,
        7,
        10,
        12,
        13,
        15,
        17,
        19,
        20,
        24,
        26,
        29,
        31,
        32,
        33,
        37,
        38,
        39,
        40,
        46,
        49,
        50,
        52,
        56,
        57,
        58,
        59,
        62,
        64,
        66,
        67,
        68,
        73,
        74,
        75,
        76,
        81,
        82,
        83,
        84,
        88,
        89,
        91,
        93,
        94,
        95,
        98,
        99,
        101,
        104,
        106,
        107,
        108,
        112,
        113,
        114,
        115,
        119,
        121,
        124,
        125,
        127,
        128,
        130,
        131,
        138,
        140,
        141,
        142,
        143,
        148,
        149,
        150,
        151,
        152,
        153,
        154
    ]
        .span();
    let class_treeids: Span<usize> = array![
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ]
        .span();
    let class_weights: Span<FP16x16> = array![
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 43690, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 65536, sign: false }
    ]
        .span();
    let classlabels: Span<usize> = array![0, 1].span();
    let nodes_falsenodeids: Span<usize> = array![
        116,
        21,
        6,
        5,
        0,
        0,
        8,
        0,
        14,
        11,
        0,
        13,
        0,
        0,
        16,
        0,
        18,
        0,
        20,
        0,
        0,
        41,
        34,
        25,
        0,
        27,
        0,
        33,
        30,
        0,
        32,
        0,
        0,
        0,
        40,
        39,
        38,
        0,
        0,
        0,
        0,
        109,
        96,
        69,
        60,
        47,
        0,
        51,
        50,
        0,
        0,
        53,
        0,
        59,
        58,
        57,
        0,
        0,
        0,
        0,
        68,
        63,
        0,
        65,
        0,
        67,
        0,
        0,
        0,
        77,
        76,
        75,
        74,
        0,
        0,
        0,
        0,
        85,
        84,
        83,
        82,
        0,
        0,
        0,
        0,
        95,
        90,
        89,
        0,
        0,
        92,
        0,
        94,
        0,
        0,
        0,
        100,
        99,
        0,
        0,
        102,
        0,
        108,
        105,
        0,
        107,
        0,
        0,
        0,
        115,
        114,
        113,
        0,
        0,
        0,
        0,
        132,
        129,
        120,
        0,
        122,
        0,
        126,
        125,
        0,
        0,
        128,
        0,
        0,
        131,
        0,
        0,
        154,
        153,
        144,
        143,
        142,
        139,
        0,
        141,
        0,
        0,
        0,
        0,
        152,
        151,
        150,
        149,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ]
        .span();
    let nodes_featureids: Span<usize> = array![
        3,
        2,
        4,
        8,
        0,
        0,
        1,
        0,
        2,
        7,
        0,
        0,
        0,
        0,
        7,
        0,
        0,
        0,
        6,
        0,
        0,
        8,
        0,
        2,
        0,
        7,
        0,
        7,
        2,
        0,
        2,
        0,
        0,
        0,
        2,
        6,
        7,
        0,
        0,
        0,
        0,
        7,
        7,
        0,
        7,
        1,
        0,
        0,
        2,
        0,
        0,
        2,
        0,
        2,
        2,
        6,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        1,
        0,
        6,
        0,
        0,
        0,
        0,
        2,
        6,
        7,
        0,
        0,
        0,
        0,
        6,
        7,
        2,
        0,
        0,
        0,
        0,
        0,
        2,
        2,
        7,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        6,
        1,
        0,
        0,
        4,
        0,
        2,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        2,
        0,
        0,
        0,
        0,
        6,
        0,
        7,
        0,
        0,
        0,
        1,
        3,
        0,
        0,
        2,
        0,
        0,
        8,
        0,
        0,
        2,
        2,
        2,
        4,
        7,
        3,
        0,
        1,
        0,
        0,
        0,
        0,
        4,
        3,
        7,
        8,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ]
        .span();
    let nodes_missing_value_tracks_true: Span<usize> = array![
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ]
        .span();
    let nodes_modes: Span<NODE_MODES> = array![
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::BRANCH_LEQ,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF,
        NODE_MODES::LEAF
    ]
        .span();
    let nodes_nodeids: Span<usize> = array![
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        144,
        145,
        146,
        147,
        148,
        149,
        150,
        151,
        152,
        153,
        154
    ]
        .span();
    let nodes_treeids: Span<usize> = array![
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ]
        .span();
    let nodes_truenodeids: Span<usize> = array![
        1,
        2,
        3,
        4,
        0,
        0,
        7,
        0,
        9,
        10,
        0,
        12,
        0,
        0,
        15,
        0,
        17,
        0,
        19,
        0,
        0,
        22,
        23,
        24,
        0,
        26,
        0,
        28,
        29,
        0,
        31,
        0,
        0,
        0,
        35,
        36,
        37,
        0,
        0,
        0,
        0,
        42,
        43,
        44,
        45,
        46,
        0,
        48,
        49,
        0,
        0,
        52,
        0,
        54,
        55,
        56,
        0,
        0,
        0,
        0,
        61,
        62,
        0,
        64,
        0,
        66,
        0,
        0,
        0,
        70,
        71,
        72,
        73,
        0,
        0,
        0,
        0,
        78,
        79,
        80,
        81,
        0,
        0,
        0,
        0,
        86,
        87,
        88,
        0,
        0,
        91,
        0,
        93,
        0,
        0,
        0,
        97,
        98,
        0,
        0,
        101,
        0,
        103,
        104,
        0,
        106,
        0,
        0,
        0,
        110,
        111,
        112,
        0,
        0,
        0,
        0,
        117,
        118,
        119,
        0,
        121,
        0,
        123,
        124,
        0,
        0,
        127,
        0,
        0,
        130,
        0,
        0,
        133,
        134,
        135,
        136,
        137,
        138,
        0,
        140,
        0,
        0,
        0,
        0,
        145,
        146,
        147,
        148,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ]
        .span();
    let nodes_values: Span<FP16x16> = array![
        FP16x16 { mag: 4096, sign: false },
        FP16x16 { mag: 22937, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 49152, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 16384, sign: false },
        FP16x16 { mag: 57344, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 19660, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 8192, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 29491, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 8192, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 24576, sign: false },
        FP16x16 { mag: 42598, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 62259, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 62259, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 40960, sign: false },
        FP16x16 { mag: 24576, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 8192, sign: false },
        FP16x16 { mag: 49152, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 19660, sign: false },
        FP16x16 { mag: 45875, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 29491, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 49152, sign: false },
        FP16x16 { mag: 42598, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 36044, sign: false },
        FP16x16 { mag: 19660, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 49152, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 45875, sign: false },
        FP16x16 { mag: 29491, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 8192, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 8192, sign: false },
        FP16x16 { mag: 36044, sign: false },
        FP16x16 { mag: 58982, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 58982, sign: false },
        FP16x16 { mag: 29491, sign: false },
        FP16x16 { mag: 8192, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 45875, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 58982, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 49152, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 42598, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 45875, sign: false },
        FP16x16 { mag: 49152, sign: false },
        FP16x16 { mag: 29491, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 45875, sign: false },
        FP16x16 { mag: 8192, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 49152, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 36044, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 58982, sign: false },
        FP16x16 { mag: 49152, sign: false },
        FP16x16 { mag: 36044, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 16384, sign: false },
        FP16x16 { mag: 20480, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 49152, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 8192, sign: false },
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 0, sign: false }
    ]
        .span();
    let base_values: Option<Span<FP16x16>> = Option::None;

    let tree_ids: Span<usize> = array![0].span();
    let mut root_index: Felt252Dict<usize> = Default::default();
    root_index.insert(0, 0);
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
        .insert(3344223123784052057366048933846905716067140384361791026153972616805110454637, 5);
    node_index
        .insert(658476905110174425295568215706634733332002869979287079110965040248935650599, 6);
    node_index
        .insert(2836212335642438363012490794290757623813171043187182819737087983331902926990, 7);
    node_index
        .insert(3496601277869056110810900082189273917786762659443522403285387602989271154262, 8);
    node_index
        .insert(1249294489531540970169611621067106471309281870082955806338234725206665112557, 9);
    node_index
        .insert(2161697998033672097816961828039488190903838124365465380011173778905747857792, 10);
    node_index
        .insert(1129815197211541481934112806673325772687763881719835256646064516195041515616, 11);
    node_index
        .insert(2592593088135949192377729543480191336537305484235681164569491942155715064163, 12);
    node_index
        .insert(578223957014284909949571568465953382377214912750427143720957054706073492593, 13);
    node_index
        .insert(1645617302026197421098102802983206579163506957138012501615708926120228167528, 14);
    node_index
        .insert(2809438816810155970395166036110536928593305127049404137239671320081144123490, 15);
    node_index
        .insert(2496308528011391755709310159103918074725328650411689040761791240500618770096, 16);
    node_index
        .insert(2003594778587446957576114348312422277631766150749194167061999666337236425714, 17);
    node_index
        .insert(2215681478480673835576618830034726157921200517935329010004363713426342305479, 18);
    node_index
        .insert(3185925835074464079989752015681272863271067691852543168049845807561733691707, 19);
    node_index
        .insert(1207265836470221457484062512091666004839070622130697586496866096347024057755, 20);
    node_index
        .insert(1870230949202979679764944800468118671928852128047695497376875566624821494262, 21);
    node_index
        .insert(618060852536781954395603948693216564334274573299243914053414488061601327758, 22);
    node_index
        .insert(232760707548494477255512699093366059519467428168757247456690480397246371463, 23);
    node_index
        .insert(1617386247965480308136742715422077429967341022950306068917456849194882895900, 24);
    node_index
        .insert(654822874782506608656472905579051041410086644071534146326024101025575400153, 25);
    node_index
        .insert(525638101901638132526332140778087078272370083489998903571807698910013602668, 26);
    node_index
        .insert(3091640181556387972179279087539287892670640556085669903494551919685982442095, 27);
    node_index
        .insert(1425411460578159050163131982087304445715005458700346341117759372943452688022, 28);
    node_index
        .insert(1722933265299553894839124723076027659619615015638971980461286818493531809034, 29);
    node_index
        .insert(3325117385742592388671007840076299062858228097051060057749225651290693960897, 30);
    node_index
        .insert(1869273998012404873272699831805499731567895666937555882116307079956228100456, 31);
    node_index
        .insert(257262395234910825879033951801423835835630270967846664413154594520703929530, 32);
    node_index
        .insert(2891500475385583315757684141371327604925143655360011721762142660942782195029, 33);
    node_index
        .insert(1257459981124043271342269816753070228024611695909553991758648317372015085782, 34);
    node_index
        .insert(3573101724490615587655146760489247477770015274618159524231872921394794809579, 35);
    node_index
        .insert(2951401777594449283985541406642940553317465718696638438535370997641527993378, 36);
    node_index
        .insert(2436860863451320452900512817385686838091627966322316039332239784330434600829, 37);
    node_index
        .insert(3257977356974702770994741663931928753019715185508521958836925918758890988390, 38);
    node_index
        .insert(2741853283805093821434776875305720302351684616683152528499335618682018880592, 39);
    node_index
        .insert(514567459251558911686762246500770717674979116530125263461114578537254680672, 40);
    node_index
        .insert(2119374930171040799805795099091470687208894498354655018353474015395489390434, 41);
    node_index
        .insert(3338470191188327918255138125570464269857839379813971679216902484398948556964, 42);
    node_index
        .insert(2892272281879752543368066497063301979597320550780387266511926397533716561161, 43);
    node_index
        .insert(2855312300216814846973137837923466865382642814675378398541743368270404441020, 44);
    node_index
        .insert(3483159989811162048659069774034779954374540681397531094699912464364012442948, 45);
    node_index
        .insert(2987290998320166766043911843685118029159841654368226419198314196237253901671, 46);
    node_index
        .insert(2925128850088180758852255336587985612621894021863350117875677692518888637440, 47);
    node_index
        .insert(2816470536741550741568042622139415760794090671576940833850781679568928363263, 48);
    node_index
        .insert(117504025904364990582663097556885493352655695615775952177872159762046032741, 49);
    node_index
        .insert(2143228410294149239354901612797540167003066966910132278060626241695943498248, 50);
    node_index
        .insert(419311759585766455354017006957403420381614228026953716552023555428752798694, 51);
    node_index
        .insert(3050064038480880151202753004776919876287903442365303272956696507808448797287, 52);
    node_index
        .insert(1385347512411195789080079656286641766866442255046855963092069449745407366357, 53);
    node_index
        .insert(3070310993421490198115289431281422702215620142859327949152517372324361472619, 54);
    node_index
        .insert(2913742884576958969164113782587195202828846527657900496424141449477472273564, 55);
    node_index
        .insert(2093568472535973986606438755824580633177115509557931302974988564932601955239, 56);
    node_index
        .insert(3560543329106347446823281318204312198881533222464682017397248462954529220234, 57);
    node_index
        .insert(2258329791422139736262782239641765930569031761627249090322755566443202104242, 58);
    node_index
        .insert(780147230530856456622774510057100334628735431063744145772648079601317149643, 59);
    node_index
        .insert(2316329094783634722527635915976455864728431870713378530935487247638854220445, 60);
    node_index
        .insert(595942459003356191117553450912822964169058193996898486073017533717706655996, 61);
    node_index
        .insert(468061318535033931711585815055033307297228787991312757359512916260570188285, 62);
    node_index
        .insert(2052204235688624923559873131063770183910134013049526186717275231865702195614, 63);
    node_index
        .insert(1699955311620840869165542755053722387608345658646185648087789689690825797785, 64);
    node_index
        .insert(3374282522812564185678772854203408947562394461702303390331208821006329361123, 65);
    node_index
        .insert(2973169188135795465401576355486514117723575153845438471619715618155257254587, 66);
    node_index
        .insert(1933845760462748501896196912926633344425020928596291295340561855718789280752, 67);
    node_index
        .insert(1400206374308839959676708676217334569580738052049798766556848516900888958934, 68);
    node_index
        .insert(1440488595273849761788031183901254714714513692476890759699232177835922420051, 69);
    node_index
        .insert(1765607197782429306903827944694032984087223086461400721152786273443512274576, 70);
    node_index
        .insert(1081728107764482028110815183657783965582618309560569428049406599883158895762, 71);
    node_index
        .insert(2062101824085365476835789898002802715794623271831111740147610520210138854237, 72);
    node_index
        .insert(2074740322618091900768870458741540994849904300182495465356314088191301853065, 73);
    node_index
        .insert(3258451235037745323160669027918885172565773098482160366154412360890640013860, 74);
    node_index
        .insert(525053653813541387331907730505904505067816165493211829943994988775279102044, 75);
    node_index
        .insert(1899573658331441767985549642643113663505618738939032010935036740376062596854, 76);
    node_index
        .insert(350484224543766923071449868701665032398970313961410080649918872017849315812, 77);
    node_index
        .insert(1950842492180490337143378914485176805944281696420768035114335939818602766139, 78);
    node_index
        .insert(1404824782481446239312837894341789608778585592445990662138109764117920511709, 79);
    node_index
        .insert(362836422984951199752185473435750713386745407518736982952373985921347236081, 80);
    node_index
        .insert(946623025367211063265176586824604502073515634531788667777364911179858705558, 81);
    node_index
        .insert(2633163324000277496191816132521100721217797223993064604664039067710591734562, 82);
    node_index
        .insert(1801986104078933931671502775029170829560335045042499367678597186639133610708, 83);
    node_index
        .insert(1420697278439090953165809531316265389371075037014378922361911811337560296928, 84);
    node_index
        .insert(2818913779862691152404893285048164649343019708946413114150419613972391643833, 85);
    node_index
        .insert(2117995436013652728497840885480545729833030913486848118093758726746902541269, 86);
    node_index
        .insert(127751852951361188238686395231851222850913859197429858579312845246901369178, 87);
    node_index
        .insert(2698811633001158191033663638617437313508153976714307643233173949778419312517, 88);
    node_index
        .insert(658388282521842455588914251287531837029259203197178137902217792556456503561, 89);
    node_index
        .insert(1181527093320872098458354979612125149419384756607076935731557552577945926179, 90);
    node_index
        .insert(749436134732178646256740138670151907037714564259781780243747781475007506978, 91);
    node_index
        .insert(139527053159256821789882596124320673637475746672994443968014105962305658551, 92);
    node_index
        .insert(2256264752321707533173578319742847366660740117899562657584919346001438808295, 93);
    node_index
        .insert(1471349294215639651865069312281269029496180149092207674923855978537861742949, 94);
    node_index
        .insert(1599527610774916650758786135513735847459194869088601099692148267264507139422, 95);
    node_index
        .insert(1348925567371118538973078195838174941892601233016661969987842843098656775084, 96);
    node_index
        .insert(3255130909854220350850821724488067913492420563978595271106701962634473840914, 97);
    node_index
        .insert(1098499015810170842401428216621470177488952811780672364884710297364076372943, 98);
    node_index
        .insert(2666902303639302012507119689908308317608522901613536135678723310999647515155, 99);
    node_index
        .insert(907997515879651052705985194221621380802961721264372722705825219340461809200, 100);
    node_index
        .insert(2124360554325144308113106422635485756539471211141315552843423768396084888273, 101);
    node_index
        .insert(3598736440043009208771817410113758019876931018927260161846683440123219507147, 102);
    node_index
        .insert(1237113034722832488580561245188430373504295256910735188987019984096012001931, 103);
    node_index
        .insert(884558344049768836371555446021588200903052780339208951904957349404044037185, 104);
    node_index
        .insert(784280321344489256066716285882203121428790637989919760379274813665427427262, 105);
    node_index
        .insert(3472551952588748711709398308465335743810517871695257916614928877311914574241, 106);
    node_index
        .insert(1579363348100943961344032004617708767155021524242506190674861550786419896732, 107);
    node_index
        .insert(653576968777651719072715499492112313607520878545254037043893560183879857489, 108);
    node_index
        .insert(2633327961579170199842757290989312779085828750765842327985383652720803061926, 109);
    node_index
        .insert(3101204920253220343970782457572784926765600523633379722044614528209389590915, 110);
    node_index
        .insert(2537565394330405662800880050062241097694806466900452037378113841155978555645, 111);
    node_index
        .insert(306955559655552244989220345789093187601563118591829582730637833945761653350, 112);
    node_index
        .insert(1144065212212058748489308207801098564095305699242880891977316839573431241916, 113);
    node_index
        .insert(3478181491851418723342103101321490659650934149094649769124337426850038155270, 114);
    node_index
        .insert(3419621624676637660673415219086314486713019053519954317586073983685881930356, 115);
    node_index
        .insert(2426908011370291613447136873176769136554489197972200481728552402228021778402, 116);
    node_index
        .insert(1916122042123370178944690083048900704842269230325086549679099089416174875473, 117);
    node_index
        .insert(2057207652658215393591191155928140567561900227203223756539551876829334137660, 118);
    node_index
        .insert(2722034389703601317070746005702467061064354401688341549606678773616189196490, 119);
    node_index
        .insert(1171026027377763359814377926117880688616494219551682642535759838199732407496, 120);
    node_index
        .insert(3507234282031533800397666430789917374211847440333243952151005899337152633413, 121);
    node_index
        .insert(591003147462937848375161803108517142253138969543815135207326321181858185919, 122);
    node_index
        .insert(182069734527202013451813026473135702900640769187641767871411473365447302169, 123);
    node_index
        .insert(1195243682249232878341146428166676460720423167409013083888435705219134747702, 124);
    node_index
        .insert(1793425644853312386902998134061844248823841892125424765064687913085130719534, 125);
    node_index
        .insert(1983622665815164792580256365519803214027269990384198703315493315153573288434, 126);
    node_index
        .insert(3615973154491344159350153395208055142342062736505558158666764642048838175685, 127);
    node_index
        .insert(2751715913626909804252433699602081411293721754810298670422380863932998088133, 128);
    node_index
        .insert(186918881712189523740089713555196200069231794627360499557319265374750577226, 129);
    node_index
        .insert(696585542544434929491503209053317581175146475161262066468664234437983008675, 130);
    node_index
        .insert(4359830495913805154545225899592517767672472055784183911796827820518038513, 131);
    node_index
        .insert(2954335207058000607751727656601539819316106074875304820535376873121805433820, 132);
    node_index
        .insert(2510390039949230255082316953804013731253145558531652907601250263563528226672, 133);
    node_index
        .insert(3226995230854300551967642178527450300960499043510855212238369890580256668532, 134);
    node_index
        .insert(1620924075233065517364532267959798304439946408626316544761884056227131075831, 135);
    node_index
        .insert(1610900122192929153657761847202689179268074338802437933866337242354758101660, 136);
    node_index
        .insert(2565949095169598991903537465065584077778440646580025930326495506484329892725, 137);
    node_index
        .insert(1012362975819634411571869839734809106575285344002573666983595104659295812607, 138);
    node_index
        .insert(242312010918799555845832460483650516749990744287009628468613253461264531026, 139);
    node_index
        .insert(1104776796569046483584574115975216172161469015460244982207905888870418040487, 140);
    node_index
        .insert(3289555912992777681578950209252840071327866822704829766247386311885634446673, 141);
    node_index
        .insert(3133389957643610781371406448279843175887428913359743769920083259111437722268, 142);
    node_index
        .insert(1169918710119352022244140656086831769713178729571654411898266328562003734517, 143);
    node_index
        .insert(3592039235252149652556167686570045881877115549259769455422056097903987237819, 144);
    node_index
        .insert(2048175709145840597887667330964815895803568760936075562647625937161113445908, 145);
    node_index
        .insert(602222645962845554276438041138511866776339653340605661136009451417275008940, 146);
    node_index
        .insert(3318742320906017551291978242369663702298606650330380959683585594592748661010, 147);
    node_index
        .insert(564160996724923690963741657975239836484028160385417016805513722318839327322, 148);
    node_index
        .insert(656294390376267384135628810815504467149264887388377312825033341338166573620, 149);
    node_index
        .insert(1201592236750942207412694706123654466634588634474700675083122904145559965915, 150);
    node_index
        .insert(2141408926815137181004274624388915700231991905288681935478972043994347966006, 151);
    node_index
        .insert(1440847977042239464860406726605567303568767649154338464116083965986084755262, 152);
    node_index
        .insert(950585553138591375958592507876257987416844837045084288783892644487908218679, 153);
    node_index
        .insert(257643451533833048856069434258149588745628261389615631070776723485957908127, 154);

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

    let mut classifier: TreeEnsembleClassifier<FP16x16> = TreeEnsembleClassifier {
        ensemble,
        class_ids,
        class_nodeids,
        class_treeids,
        class_weights,
        classlabels,
        base_values,
        post_transform
    };

    let mut X = TensorTrait::new(
        array![1, 9].span(),
        array![
            FP16x16 { mag: 39321, sign: false },
            FP16x16 { mag: 32768, sign: false },
            FP16x16 { mag: 52428, sign: false },
            FP16x16 { mag: 16384, sign: false },
            FP16x16 { mag: 0, sign: false },
            FP16x16 { mag: 65536, sign: false },
            FP16x16 { mag: 0, sign: false },
            FP16x16 { mag: 16384, sign: false },
            FP16x16 { mag: 0, sign: false },
        ]
            .span()
    );

    (classifier, X)
}
