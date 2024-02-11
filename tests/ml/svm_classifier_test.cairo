use orion::numbers::FP16x16;
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, U32Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP16x16TensorPartialEq;

use orion::numbers::FP64x64;
use orion::operators::tensor::implementations::tensor_fp64x64::{
    FP64x64Tensor, FP64x64TensorPartialEq
};

use orion::operators::ml::svm::svm_classifier::{SVMClassifierTrait, POST_TRANSFORM, SVMClassifier};
use orion::operators::ml::svm::core::{KERNEL_TYPE};


#[test]
#[available_gas(200000000000)]
fn test_svm_classifier_noprob_linear_sv_none() {
    let post_transform = POST_TRANSFORM::NONE;
    let (mut classifier, X) = svm_classifier_binary_noprob_linear_sv(post_transform);

    let (labels, scores) = SVMClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 0, 'labels[0]');
    assert(*labels[1] == 0, 'labels[1]');
    assert(*labels[2] == 0, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    let mut expected_scores: Tensor<FP16x16> = TensorTrait::new(
        array![3, 2].span(),
        array![
            FP16x16 { mag: 174499, sign: true },
            FP16x16 { mag: 174499, sign: false },
            FP16x16 { mag: 145149, sign: true },
            FP16x16 { mag: 145149, sign: false },
            FP16x16 { mag: 115799, sign: true },
            FP16x16 { mag: 115799, sign: false }
        ]
            .span()
    );

    assert_eq(scores, expected_scores);
}


#[test]
#[available_gas(200000000000)]
fn test_svm_classifier_noprob_linear_sv_logistic() {
    let post_transform = POST_TRANSFORM::LOGISTIC;
    let (mut classifier, X) = svm_classifier_binary_noprob_linear_sv(post_transform);

    let (labels, scores) = SVMClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 0, 'labels[0]');
    assert(*labels[1] == 0, 'labels[1]');
    assert(*labels[2] == 0, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    let mut expected_scores: Tensor<FP16x16> = TensorTrait::new(
        array![3, 2].span(),
        array![
            FP16x16 { mag: 4273, sign: false },
            FP16x16 { mag: 61262, sign: false },
            FP16x16 { mag: 6450, sign: false },
            FP16x16 { mag: 59085, sign: false },
            FP16x16 { mag: 9563, sign: false },
            FP16x16 { mag: 55972, sign: false }
        ]
            .span()
    );

    assert_eq(scores, expected_scores);
}

#[test]
#[available_gas(200000000000)]
fn test_svm_classifier_noprob_linear_sv_softmax() {
    let post_transform = POST_TRANSFORM::SOFTMAX;
    let (mut classifier, X) = svm_classifier_binary_noprob_linear_sv(post_transform);

    let (labels, scores) = SVMClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 0, 'labels[0]');
    assert(*labels[1] == 0, 'labels[1]');
    assert(*labels[2] == 0, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    let mut expected_scores: Tensor<FP16x16> = TensorTrait::new(
        array![3, 2].span(),
        array![
            FP16x16 { mag: 317, sign: false },
            FP16x16 { mag: 65218, sign: false },
            FP16x16 { mag: 771, sign: false },
            FP16x16 { mag: 64764, sign: false },
            FP16x16 { mag: 1858, sign: false },
            FP16x16 { mag: 63677, sign: false }
        ]
            .span()
    );

    assert_eq(scores, expected_scores);
}

#[test]
#[available_gas(200000000000)]
fn test_svm_classifier_noprob_linear_sv_softmax_zero() {
    let post_transform = POST_TRANSFORM::SOFTMAXZERO;
    let (mut classifier, X) = svm_classifier_binary_noprob_linear_sv(post_transform);

    let (labels, scores) = SVMClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 0, 'labels[0]');
    assert(*labels[1] == 0, 'labels[1]');
    assert(*labels[2] == 0, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    let mut expected_scores: Tensor<FP16x16> = TensorTrait::new(
        array![3, 2].span(),
        array![
            FP16x16 { mag: 317, sign: false },
            FP16x16 { mag: 65218, sign: false },
            FP16x16 { mag: 771, sign: false },
            FP16x16 { mag: 64764, sign: false },
            FP16x16 { mag: 1858, sign: false },
            FP16x16 { mag: 63677, sign: false }
        ]
            .span()
    );

    assert_eq(scores, expected_scores);
}


#[test]
#[available_gas(200000000000)]
fn test_svm_classifier_noprob_linear_none() {
    let post_transform = POST_TRANSFORM::NONE;
    let (mut classifier, X) = svm_classifier_helper_noprob_linear(post_transform);

    let (labels, scores) = SVMClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 2, 'labels[0]');
    assert(*labels[1] == 3, 'labels[1]');
    assert(*labels[2] == 0, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    let mut expected_scores: Tensor<FP16x16> = TensorTrait::new(
        array![3, 4].span(),
        array![
            FP16x16 { mag: 7738, sign: true },
            FP16x16 { mag: 29929, sign: true },
            FP16x16 { mag: 27248, sign: false },
            FP16x16 { mag: 21922, sign: false },
            FP16x16 { mag: 4021, sign: true },
            FP16x16 { mag: 15167, sign: true },
            FP16x16 { mag: 4843, sign: false },
            FP16x16 { mag: 5979, sign: false },
            FP16x16 { mag: 304, sign: true },
            FP16x16 { mag: 406, sign: true },
            FP16x16 { mag: 17562, sign: true },
            FP16x16 { mag: 9962, sign: true },
        ]
            .span()
    );

    assert_eq(scores, expected_scores);
}


#[test]
#[available_gas(200000000000)]
fn test_svm_classifier_noprob_linear_logistic() {
    let post_transform = POST_TRANSFORM::LOGISTIC;
    let (mut classifier, X) = svm_classifier_helper_noprob_linear(post_transform);

    let (labels, scores) = SVMClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 2, 'labels[0]');
    assert(*labels[1] == 3, 'labels[1]');
    assert(*labels[2] == 0, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    let mut expected_scores: Tensor<FP16x16> = TensorTrait::new(
        array![3, 4].span(),
        array![
            FP16x16 { mag: 30835, sign: false },
            FP16x16 { mag: 25413, sign: false },
            FP16x16 { mag: 39483, sign: false },
            FP16x16 { mag: 38197, sign: false },
            FP16x16 { mag: 31762, sign: false },
            FP16x16 { mag: 28992, sign: false },
            FP16x16 { mag: 33978, sign: false },
            FP16x16 { mag: 34261, sign: false },
            FP16x16 { mag: 32691, sign: false },
            FP16x16 { mag: 32666, sign: false },
            FP16x16 { mag: 28403, sign: false },
            FP16x16 { mag: 30282, sign: false }
        ]
            .span()
    );

    assert_eq(scores, expected_scores);
}


#[test]
#[available_gas(200000000000)]
fn test_svm_classifier_noprob_linear_softmax() {
    let post_transform = POST_TRANSFORM::SOFTMAX;
    let (mut classifier, X) = svm_classifier_helper_noprob_linear(post_transform);

    let (labels, scores) = SVMClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 2, 'labels[0]');
    assert(*labels[1] == 3, 'labels[1]');
    assert(*labels[2] == 0, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    let mut expected_scores: Tensor<FP16x16> = TensorTrait::new(
        array![3, 4].span(),
        array![
            FP16x16 { mag: 13131, sign: false },
            FP16x16 { mag: 9359, sign: false },
            FP16x16 { mag: 22396, sign: false },
            FP16x16 { mag: 20648, sign: false },
            FP16x16 { mag: 15779, sign: false },
            FP16x16 { mag: 13311, sign: false },
            FP16x16 { mag: 18064, sign: false },
            FP16x16 { mag: 18380, sign: false },
            FP16x16 { mag: 18054, sign: false },
            FP16x16 { mag: 18026, sign: false },
            FP16x16 { mag: 13874, sign: false },
            FP16x16 { mag: 15580, sign: false },
        ]
            .span()
    );

    assert_eq(scores, expected_scores);
}

#[test]
#[available_gas(200000000000)]
fn test_svm_classifier_noprob_linear_softmax_zero() {
    let post_transform = POST_TRANSFORM::SOFTMAXZERO;
    let (mut classifier, X) = svm_classifier_helper_noprob_linear(post_transform);

    let (labels, scores) = SVMClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 2, 'labels[0]');
    assert(*labels[1] == 3, 'labels[1]');
    assert(*labels[2] == 0, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    let mut expected_scores: Tensor<FP16x16> = TensorTrait::new(
        array![3, 4].span(),
        array![
            FP16x16 { mag: 13131, sign: false },
            FP16x16 { mag: 9359, sign: false },
            FP16x16 { mag: 22396, sign: false },
            FP16x16 { mag: 20648, sign: false },
            FP16x16 { mag: 15779, sign: false },
            FP16x16 { mag: 13311, sign: false },
            FP16x16 { mag: 18064, sign: false },
            FP16x16 { mag: 18380, sign: false },
            FP16x16 { mag: 18054, sign: false },
            FP16x16 { mag: 18026, sign: false },
            FP16x16 { mag: 13874, sign: false },
            FP16x16 { mag: 15580, sign: false },
        ]
            .span()
    );

    assert_eq(scores, expected_scores);
}

#[test]
#[available_gas(200000000000)]
fn test_svm_classifier_linear_none() {
    let post_transform = POST_TRANSFORM::NONE;
    let (mut classifier, X) = svm_classifier_helper_linear(post_transform);

    let (labels, scores) = SVMClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 2, 'labels[0]');
    assert(*labels[1] == 3, 'labels[1]');
    assert(*labels[2] == 0, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    let mut expected_scores: Tensor<FP16x16> = TensorTrait::new(
        array![3, 4].span(),
        array![
            FP16x16 { mag: 7738, sign: true },
            FP16x16 { mag: 29929, sign: true },
            FP16x16 { mag: 27248, sign: false },
            FP16x16 { mag: 21922, sign: false },
            FP16x16 { mag: 4021, sign: true },
            FP16x16 { mag: 15167, sign: true },
            FP16x16 { mag: 4843, sign: false },
            FP16x16 { mag: 5979, sign: false },
            FP16x16 { mag: 304, sign: true },
            FP16x16 { mag: 406, sign: true },
            FP16x16 { mag: 17562, sign: true },
            FP16x16 { mag: 9962, sign: true },
        ]
            .span()
    );

    assert_eq(scores, expected_scores);
}

#[test]
#[available_gas(200000000000)]
fn test_svm_classifier_linear_logistic() {
    let post_transform = POST_TRANSFORM::LOGISTIC;
    let (mut classifier, X) = svm_classifier_helper_linear(post_transform);

    let (labels, scores) = SVMClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 2, 'labels[0]');
    assert(*labels[1] == 3, 'labels[1]');
    assert(*labels[2] == 0, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    let mut expected_scores: Tensor<FP16x16> = TensorTrait::new(
        array![3, 4].span(),
        array![
            FP16x16 { mag: 30835, sign: false },
            FP16x16 { mag: 25413, sign: false },
            FP16x16 { mag: 39483, sign: false },
            FP16x16 { mag: 38197, sign: false },
            FP16x16 { mag: 31762, sign: false },
            FP16x16 { mag: 28992, sign: false },
            FP16x16 { mag: 33978, sign: false },
            FP16x16 { mag: 34261, sign: false },
            FP16x16 { mag: 32691, sign: false },
            FP16x16 { mag: 32666, sign: false },
            FP16x16 { mag: 28403, sign: false },
            FP16x16 { mag: 30282, sign: false }
        ]
            .span()
    );

    assert_eq(scores, expected_scores);
}

#[test]
#[available_gas(200000000000)]
fn test_svm_classifier_linear_softmax() {
    let post_transform = POST_TRANSFORM::SOFTMAX;
    let (mut classifier, X) = svm_classifier_helper_linear(post_transform);

    let (labels, scores) = SVMClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 2, 'labels[0]');
    assert(*labels[1] == 3, 'labels[1]');
    assert(*labels[2] == 0, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    let mut expected_scores: Tensor<FP16x16> = TensorTrait::new(
        array![3, 4].span(),
        array![
            FP16x16 { mag: 13131, sign: false },
            FP16x16 { mag: 9359, sign: false },
            FP16x16 { mag: 22396, sign: false },
            FP16x16 { mag: 20648, sign: false },
            FP16x16 { mag: 15779, sign: false },
            FP16x16 { mag: 13311, sign: false },
            FP16x16 { mag: 18064, sign: false },
            FP16x16 { mag: 18380, sign: false },
            FP16x16 { mag: 18054, sign: false },
            FP16x16 { mag: 18026, sign: false },
            FP16x16 { mag: 13874, sign: false },
            FP16x16 { mag: 15580, sign: false },
        ]
            .span()
    );

    assert_eq(scores, expected_scores);
}


#[test]
#[available_gas(200000000000)]
fn test_svm_classifier_linear_softmax_zero() {
    let post_transform = POST_TRANSFORM::SOFTMAXZERO;
    let (mut classifier, X) = svm_classifier_helper_linear(post_transform);

    let (labels, scores) = SVMClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 2, 'labels[0]');
    assert(*labels[1] == 3, 'labels[1]');
    assert(*labels[2] == 0, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    let mut expected_scores: Tensor<FP16x16> = TensorTrait::new(
        array![3, 4].span(),
        array![
            FP16x16 { mag: 13131, sign: false },
            FP16x16 { mag: 9359, sign: false },
            FP16x16 { mag: 22396, sign: false },
            FP16x16 { mag: 20648, sign: false },
            FP16x16 { mag: 15779, sign: false },
            FP16x16 { mag: 13311, sign: false },
            FP16x16 { mag: 18064, sign: false },
            FP16x16 { mag: 18380, sign: false },
            FP16x16 { mag: 18054, sign: false },
            FP16x16 { mag: 18026, sign: false },
            FP16x16 { mag: 13874, sign: false },
            FP16x16 { mag: 15580, sign: false },
        ]
            .span()
    );

    assert_eq(scores, expected_scores);
}


#[test]
#[available_gas(200000000000)]
fn test_svm_classifier_binary_none_fp64x64() {
    let post_transform = POST_TRANSFORM::NONE;
    let (mut classifier, X) = svm_classifier_helper_fp64x64(post_transform);

    let (labels, scores) = SVMClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 0, 'labels[0]');
    assert(*labels[1] == 1, 'labels[1]');
    assert(*labels[2] == 1, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    let mut expected_scores: Tensor<FP64x64> = TensorTrait::new(
        array![3, 2].span(),
        array![
            FP64x64 { mag: 18322911080742739968, sign: false },
            FP64x64 { mag: 123832992966812224, sign: false },
            FP64x64 { mag: 8658920114943337472, sign: false },
            FP64x64 { mag: 9787823958766215168, sign: false },
            FP64x64 { mag: 276645820873422144, sign: false },
            FP64x64 { mag: 18170098252836128768, sign: false }
        ]
            .span()
    );

    assert_eq(scores, expected_scores);
}


#[test]
#[available_gas(200000000000)]
fn test_svm_classifier_binary_logistic_fp64x64() {
    let post_transform = POST_TRANSFORM::LOGISTIC;
    let (mut classifier, X) = svm_classifier_helper_fp64x64(post_transform);

    let (labels, scores) = SVMClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 0, 'labels[0]');
    assert(*labels[1] == 1, 'labels[1]');
    assert(*labels[2] == 1, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    let mut expected_scores: Tensor<FP64x64> = TensorTrait::new(
        array![3, 2].span(),
        array![
            FP64x64 { mag: 13461271680116586496, sign: false },
            FP64x64 { mag: 9254325673410459648, sign: false },
            FP64x64 { mag: 11349211717397211136, sign: false },
            FP64x64 { mag: 11614494343921229824, sign: false },
            FP64x64 { mag: 9292528880387112960, sign: false },
            FP64x64 { mag: 13431074360067923968, sign: false }
        ]
            .span()
    );

    assert_eq(scores, expected_scores);
}

#[test]
#[available_gas(200000000000)]
fn test_svm_classifier_binary_softmax_fp64x64() {
    let post_transform = POST_TRANSFORM::SOFTMAX;
    let (mut classifier, X) = svm_classifier_helper_fp64x64(post_transform);

    let (labels, scores) = SVMClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 0, 'labels[0]');
    assert(*labels[1] == 1, 'labels[1]');
    assert(*labels[2] == 1, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    let mut expected_scores: Tensor<FP64x64> = TensorTrait::new(
        array![3, 2].span(),
        array![
            FP64x64 { mag: 13436811297474848768, sign: false },
            FP64x64 { mag: 5009932776234703872, sign: false },
            FP64x64 { mag: 8941229086247388160, sign: false },
            FP64x64 { mag: 9505514987462162432, sign: false },
            FP64x64 { mag: 5070622564237207552, sign: false },
            FP64x64 { mag: 13376121509472344064, sign: false }
        ]
            .span()
    );

    assert_eq(scores, expected_scores);
}

#[test]
#[available_gas(200000000000)]
fn test_svm_classifier_binary_softmax_zero_fp64x64() {
    let post_transform = POST_TRANSFORM::SOFTMAXZERO;
    let (mut classifier, X) = svm_classifier_helper_fp64x64(post_transform);

    let (labels, scores) = SVMClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 0, 'labels[0]');
    assert(*labels[1] == 1, 'labels[1]');
    assert(*labels[2] == 1, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    let mut expected_scores: Tensor<FP64x64> = TensorTrait::new(
        array![3, 2].span(),
        array![
            FP64x64 { mag: 13436811297474848768, sign: false },
            FP64x64 { mag: 5009932776234703872, sign: false },
            FP64x64 { mag: 8941229086247388160, sign: false },
            FP64x64 { mag: 9505514987462162432, sign: false },
            FP64x64 { mag: 5070622564237207552, sign: false },
            FP64x64 { mag: 13376121509472344064, sign: false }
        ]
            .span()
    );

    assert_eq(scores, expected_scores);
}


// ============ HELPER ============ //

fn svm_classifier_helper_linear(
    post_transform: POST_TRANSFORM
) -> (SVMClassifier<FP16x16>, Tensor<FP16x16>) {
    let coefficients: Span<FP16x16> = array![
        FP16x16 { mag: 10169, sign: true },
        FP16x16 { mag: 15905, sign: false },
        FP16x16 { mag: 459, sign: false },
        FP16x16 { mag: 26713, sign: false },
        FP16x16 { mag: 2129, sign: true },
        FP16x16 { mag: 18, sign: false },
        FP16x16 { mag: 12830, sign: true },
        FP16x16 { mag: 23097, sign: true },
        FP16x16 { mag: 1415, sign: true },
        FP16x16 { mag: 28717, sign: true },
        FP16x16 { mag: 2994, sign: false },
        FP16x16 { mag: 847, sign: true }
    ]
        .span();
    let kernel_params: Span<FP16x16> = array![
        FP16x16 { mag: 65, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 196608, sign: false }
    ]
        .span();
    let kernel_type = KERNEL_TYPE::LINEAR;
    let prob_a: Span<FP16x16> = array![FP16x16 { mag: 336797, sign: true }].span();
    let prob_b: Span<FP16x16> = array![FP16x16 { mag: 4194, sign: false }].span();
    let rho: Span<FP16x16> = array![
        FP16x16 { mag: 4908, sign: true },
        FP16x16 { mag: 11563, sign: true },
        FP16x16 { mag: 13872, sign: true },
        FP16x16 { mag: 33829, sign: true }
    ]
        .span();

    let support_vectors: Span<FP16x16> = array![].span();
    let classlabels: Span<usize> = array![0, 1, 2, 3].span();

    let vectors_per_class = Option::None;

    let mut classifier: SVMClassifier<FP16x16> = SVMClassifier {
        classlabels,
        coefficients,
        kernel_params,
        kernel_type,
        post_transform,
        prob_a,
        prob_b,
        rho,
        support_vectors,
        vectors_per_class,
    };

    let mut X: Tensor<FP16x16> = TensorTrait::new(
        array![3, 3].span(),
        array![
            FP16x16 { mag: 65536, sign: true },
            FP16x16 { mag: 52428, sign: true },
            FP16x16 { mag: 39321, sign: true },
            FP16x16 { mag: 26214, sign: true },
            FP16x16 { mag: 13107, sign: true },
            FP16x16 { mag: 0, sign: false },
            FP16x16 { mag: 13107, sign: false },
            FP16x16 { mag: 26214, sign: false },
            FP16x16 { mag: 39321, sign: false },
        ]
            .span()
    );

    (classifier, X)
}


fn svm_classifier_binary_noprob_linear_sv(
    post_transform: POST_TRANSFORM
) -> (SVMClassifier<FP16x16>, Tensor<FP16x16>) {
    let coefficients: Span<FP16x16> = array![
        FP16x16 { mag: 50226, sign: false },
        FP16x16 { mag: 5711, sign: false },
        FP16x16 { mag: 7236, sign: false },
        FP16x16 { mag: 63175, sign: true }
    ]
        .span();
    let kernel_params: Span<FP16x16> = array![
        FP16x16 { mag: 8025, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 196608, sign: false }
    ]
        .span();
    let kernel_type = KERNEL_TYPE::LINEAR;
    let prob_a: Span<FP16x16> = array![].span();
    let prob_b: Span<FP16x16> = array![].span();
    let rho: Span<FP16x16> = array![FP16x16 { mag: 146479, sign: false }].span();

    let support_vectors: Span<FP16x16> = array![
        FP16x16 { mag: 314572, sign: false },
        FP16x16 { mag: 222822, sign: false },
        FP16x16 { mag: 124518, sign: false },
        FP16x16 { mag: 327680, sign: false },
        FP16x16 { mag: 196608, sign: false },
        FP16x16 { mag: 104857, sign: false },
        FP16x16 { mag: 294912, sign: false },
        FP16x16 { mag: 150732, sign: false },
        FP16x16 { mag: 85196, sign: false },
        FP16x16 { mag: 334233, sign: false },
        FP16x16 { mag: 163840, sign: false },
        FP16x16 { mag: 196608, sign: false }
    ]
        .span();
    let classlabels: Span<usize> = array![0, 1].span();

    let vectors_per_class = Option::Some(array![3, 1].span());

    let mut classifier: SVMClassifier<FP16x16> = SVMClassifier {
        classlabels,
        coefficients,
        kernel_params,
        kernel_type,
        post_transform,
        prob_a,
        prob_b,
        rho,
        support_vectors,
        vectors_per_class,
    };

    let mut X: Tensor<FP16x16> = TensorTrait::new(
        array![3, 3].span(),
        array![
            FP16x16 { mag: 65536, sign: true },
            FP16x16 { mag: 52428, sign: true },
            FP16x16 { mag: 39321, sign: true },
            FP16x16 { mag: 26214, sign: true },
            FP16x16 { mag: 13107, sign: true },
            FP16x16 { mag: 0, sign: false },
            FP16x16 { mag: 13107, sign: false },
            FP16x16 { mag: 26214, sign: false },
            FP16x16 { mag: 39321, sign: false },
        ]
            .span()
    );

    (classifier, X)
}


fn svm_classifier_helper_noprob_linear(
    post_transform: POST_TRANSFORM
) -> (SVMClassifier<FP16x16>, Tensor<FP16x16>) {
    let coefficients: Span<FP16x16> = array![
        FP16x16 { mag: 10169, sign: true },
        FP16x16 { mag: 15905, sign: false },
        FP16x16 { mag: 459, sign: false },
        FP16x16 { mag: 26713, sign: false },
        FP16x16 { mag: 2129, sign: true },
        FP16x16 { mag: 18, sign: false },
        FP16x16 { mag: 12830, sign: true },
        FP16x16 { mag: 23097, sign: true },
        FP16x16 { mag: 1415, sign: true },
        FP16x16 { mag: 28717, sign: true },
        FP16x16 { mag: 2994, sign: false },
        FP16x16 { mag: 847, sign: true }
    ]
        .span();
    let kernel_params: Span<FP16x16> = array![
        FP16x16 { mag: 65, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 196608, sign: false }
    ]
        .span();
    let kernel_type = KERNEL_TYPE::LINEAR;
    let prob_a: Span<FP16x16> = array![].span();
    let prob_b: Span<FP16x16> = array![].span();
    let rho: Span<FP16x16> = array![
        FP16x16 { mag: 4908, sign: true },
        FP16x16 { mag: 11563, sign: true },
        FP16x16 { mag: 13872, sign: true },
        FP16x16 { mag: 33829, sign: true }
    ]
        .span();

    let support_vectors: Span<FP16x16> = array![].span();
    let classlabels: Span<usize> = array![0, 1, 2, 3].span();

    let vectors_per_class = Option::None;

    let mut classifier: SVMClassifier<FP16x16> = SVMClassifier {
        classlabels,
        coefficients,
        kernel_params,
        kernel_type,
        post_transform,
        prob_a,
        prob_b,
        rho,
        support_vectors,
        vectors_per_class,
    };

    let mut X: Tensor<FP16x16> = TensorTrait::new(
        array![3, 3].span(),
        array![
            FP16x16 { mag: 65536, sign: true },
            FP16x16 { mag: 52428, sign: true },
            FP16x16 { mag: 39321, sign: true },
            FP16x16 { mag: 26214, sign: true },
            FP16x16 { mag: 13107, sign: true },
            FP16x16 { mag: 0, sign: false },
            FP16x16 { mag: 13107, sign: false },
            FP16x16 { mag: 26214, sign: false },
            FP16x16 { mag: 39321, sign: false },
        ]
            .span()
    );

    (classifier, X)
}


fn svm_classifier_helper_fp64x64(
    post_transform: POST_TRANSFORM
) -> (SVMClassifier<FP64x64>, Tensor<FP64x64>) {
    let coefficients: Span<FP64x64> = array![
        FP64x64 { mag: 18446744073709551616, sign: false },
        FP64x64 { mag: 18446744073709551616, sign: false },
        FP64x64 { mag: 18446744073709551616, sign: false },
        FP64x64 { mag: 18446744073709551616, sign: false },
        FP64x64 { mag: 18446744073709551616, sign: true },
        FP64x64 { mag: 18446744073709551616, sign: true },
        FP64x64 { mag: 18446744073709551616, sign: true },
        FP64x64 { mag: 18446744073709551616, sign: true }
    ]
        .span();
    let kernel_params: Span<FP64x64> = array![
        FP64x64 { mag: 7054933896252620800, sign: false },
        FP64x64 { mag: 0, sign: false },
        FP64x64 { mag: 55340232221128654848, sign: false }
    ]
        .span();
    let kernel_type = KERNEL_TYPE::RBF;
    let prob_a: Span<FP64x64> = array![FP64x64 { mag: 94799998099962986496, sign: true }].span();
    let prob_b: Span<FP64x64> = array![FP64x64 { mag: 1180576833385529344, sign: false }].span();
    let rho: Span<FP64x64> = array![FP64x64 { mag: 3082192501545631744, sign: false }].span();

    let support_vectors: Span<FP64x64> = array![
        FP64x64 { mag: 3528081300248330240, sign: false },
        FP64x64 { mag: 19594207602596118528, sign: true },
        FP64x64 { mag: 9235613999318433792, sign: false },
        FP64x64 { mag: 10869715877100519424, sign: true },
        FP64x64 { mag: 5897111318564962304, sign: true },
        FP64x64 { mag: 1816720038917308416, sign: false },
        FP64x64 { mag: 4564890528671334400, sign: false },
        FP64x64 { mag: 21278987070814027776, sign: true },
        FP64x64 { mag: 7581529597213147136, sign: false },
        FP64x64 { mag: 10953113834067329024, sign: true },
        FP64x64 { mag: 24318984989010034688, sign: true },
        FP64x64 { mag: 30296187483321270272, sign: true },
        FP64x64 { mag: 10305112258191032320, sign: false },
        FP64x64 { mag: 17005441559857987584, sign: true },
        FP64x64 { mag: 11555205301925838848, sign: false },
        FP64x64 { mag: 2962701975885447168, sign: true },
        FP64x64 { mag: 11741665981322231808, sign: true },
        FP64x64 { mag: 15376232508819505152, sign: false },
        FP64x64 { mag: 13908474645692022784, sign: false },
        FP64x64 { mag: 7323415394302033920, sign: true },
        FP64x64 { mag: 3284258824352956416, sign: true },
        FP64x64 { mag: 11374683084831064064, sign: true },
        FP64x64 { mag: 9087138148126818304, sign: false },
        FP64x64 { mag: 8247488946750095360, sign: false }
    ]
        .span();
    let classlabels: Span<usize> = array![0, 1].span();

    let vectors_per_class = Option::Some(array![4, 4].span());

    let mut classifier: SVMClassifier<FP64x64> = SVMClassifier {
        classlabels,
        coefficients,
        kernel_params,
        kernel_type,
        post_transform,
        prob_a,
        prob_b,
        rho,
        support_vectors,
        vectors_per_class,
    };

    let mut X: Tensor<FP64x64> = TensorTrait::new(
        array![3, 3].span(),
        array![
            FP64x64 { mag: 18446744073709551616, sign: true },
            FP64x64 { mag: 14757395258967642112, sign: true },
            FP64x64 { mag: 11068046444225730560, sign: true },
            FP64x64 { mag: 7378697629483821056, sign: true },
            FP64x64 { mag: 3689348814741910528, sign: true },
            FP64x64 { mag: 0, sign: false },
            FP64x64 { mag: 3689348814741910528, sign: false },
            FP64x64 { mag: 7378697629483821056, sign: false },
            FP64x64 { mag: 11068046444225730560, sign: false }
        ]
            .span()
    );

    (classifier, X)
}

