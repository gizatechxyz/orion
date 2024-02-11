use orion::numbers::FP16x16;
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, U32Tensor};

use orion::operators::ml::linear::linear_classifier::{
    LinearClassifierTrait, POST_TRANSFORM, LinearClassifier
};
use core::debug::PrintTrait;

#[test]
#[available_gas(200000000000)]
fn test_linear_classifier_multi_none() {
    let (mut classifier, X) = linear_classifier_helper(POST_TRANSFORM::NONE);

    let (labels, mut scores) = LinearClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 0, 'labels[0]');
    assert(*labels[1] == 2, 'labels[1]');
    assert(*labels[2] == 2, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    assert(*scores.data[0] == FP16x16 { mag: 157942, sign: false }, '*scores[0] == 2.41');
    assert(*scores.data[1] == FP16x16 { mag: 138936, sign: true }, '*scores[1] == -2.12');
    assert(*scores.data[2] == FP16x16 { mag: 38666, sign: false }, '*scores[2] == 0.59');
    assert(*scores.data[3] == FP16x16 { mag: 43910, sign: false }, '*scores[3] == 0.67');
    assert(*scores.data[4] == FP16x16 { mag: 74710, sign: true }, '*scores[4] == -1.14');
    assert(*scores.data[5] == FP16x16 { mag: 88472, sign: false }, '*scores[5] == 1.35');
    assert(*scores.data[6] == FP16x16 { mag: 70122, sign: true }, '*scores[6] == -1.07');
    assert(*scores.data[7] == FP16x16 { mag: 10484, sign: true }, '*scores[7] == -0.16');
    assert(*scores.data[8] == FP16x16 { mag: 138278, sign: false }, '*scores[8] == 2.11');
}


#[test]
#[available_gas(200000000000)]
fn test_linear_classifier_multi_softmax() {
    let (mut classifier, X) = linear_classifier_helper(POST_TRANSFORM::SOFTMAX);

    let (labels, mut scores) = LinearClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 0, 'labels[0]');
    assert(*labels[1] == 2, 'labels[1]');
    assert(*labels[2] == 2, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    assert(*scores.data[0] == FP16x16 { mag: 55879, sign: false }, '*scores[0] == 0.852656');
    assert(*scores.data[1] == FP16x16 { mag: 602, sign: false }, '*scores[1] == 0.009192');
    assert(*scores.data[2] == FP16x16 { mag: 9053, sign: false }, '*scores[2] == 0.138152');
    assert(*scores.data[3] == FP16x16 { mag: 20888, sign: false }, '*scores[3] == 0.318722');
    assert(*scores.data[4] == FP16x16 { mag: 3418, sign: false }, '*scores[4] == 0.05216');
    assert(*scores.data[5] == FP16x16 { mag: 41229, sign: false }, '*scores[5] == 0.629118');
    assert(*scores.data[6] == FP16x16 { mag: 2380, sign: false }, '*scores[6] == 0.036323');
    assert(*scores.data[7] == FP16x16 { mag: 5914, sign: false }, '*scores[7] == 0.090237');
    assert(*scores.data[8] == FP16x16 { mag: 57241, sign: false }, '*scores[8] == 0.87344');
}

#[test]
#[available_gas(200000000000)]
fn test_linear_classifier_multi_softmax_zero() {
    let (mut classifier, X) = linear_classifier_helper(POST_TRANSFORM::SOFTMAXZERO);

    let (labels, mut scores) = LinearClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 0, 'labels[0]');
    assert(*labels[1] == 2, 'labels[1]');
    assert(*labels[2] == 2, 'labels[2]');
    assert(labels.len() == 3, 'len(labels)');

    // ASSERT SCORES
    assert(*scores.data[0] == FP16x16 { mag: 55879, sign: false }, '*scores[0] == 0.852656');
    assert(*scores.data[1] == FP16x16 { mag: 602, sign: false }, '*scores[1] == 0.009192');
    assert(*scores.data[2] == FP16x16 { mag: 9053, sign: false }, '*scores[2] == 0.138152');
    assert(*scores.data[3] == FP16x16 { mag: 20888, sign: false }, '*scores[3] == 0.318722');
    assert(*scores.data[4] == FP16x16 { mag: 3418, sign: false }, '*scores[4] == 0.05216');
    assert(*scores.data[5] == FP16x16 { mag: 41229, sign: false }, '*scores[5] == 0.629118');
    assert(*scores.data[6] == FP16x16 { mag: 2380, sign: false }, '*scores[6] == 0.036323');
    assert(*scores.data[7] == FP16x16 { mag: 5914, sign: false }, '*scores[7] == 0.090237');
    assert(*scores.data[8] == FP16x16 { mag: 57241, sign: false }, '*scores[8] == 0.87344');
}


#[test]
#[available_gas(200000000000)]
fn test_linear_classifier_multi_logistic() {
    let (mut classifier, X) = linear_classifier_helper(POST_TRANSFORM::LOGISTIC);

    let (labels, mut scores) = LinearClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 0, 'labels[0] == 0');
    assert(*labels[1] == 2, 'labels[1] == 2');
    assert(*labels[2] == 2, 'labels[2] == 2');
    assert(labels.len() == 3, 'len(labels) == 3');

    // ASSERT SCORES
    assert(*scores.data[0] == FP16x16 { mag: 60135, sign: false }, '*scores[0] == 0.917587');
    assert(*scores.data[1] == FP16x16 { mag: 7023, sign: false }, '*scores[1] == 0.107168');
    assert(*scores.data[2] == FP16x16 { mag: 42163, sign: false }, '*scores[2] == 0.643365');
    assert(*scores.data[3] == FP16x16 { mag: 43351, sign: false }, '*scores[3] == 0.661503');
    assert(*scores.data[4] == FP16x16 { mag: 15881, sign: false }, '*scores[4] == 0.24232');
    assert(*scores.data[5] == FP16x16 { mag: 52043, sign: false }, '*scores[5] == 0.79413');
    assert(*scores.data[6] == FP16x16 { mag: 16738, sign: false }, '*scores[6] == 0.255403');
    assert(*scores.data[7] == FP16x16 { mag: 30152, sign: false }, '*scores[7] == 0.460085');
    assert(*scores.data[8] == FP16x16 { mag: 58450, sign: false }, '*scores[8] == 0.891871');
}

#[test]
#[available_gas(200000000000)]
fn test_linear_classifier_binary_none() {
    let (mut classifier, X) = linear_classifier_helper_binary(POST_TRANSFORM::NONE);

    let (labels, mut scores) = LinearClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 1, 'labels[0]');
    assert(*labels[1] == 1, 'labels[1]');
    assert(labels.len() == 2, 'len(labels)');

    // ASSERT SCORES
    assert(*scores.data[0] == FP16x16 { mag: 624559, sign: true }, '*scores[0] == -9.53');
    assert(*scores.data[1] == FP16x16 { mag: 624559, sign: false }, '*scores[1] == 9.53');
    assert(*scores.data[2] == FP16x16 { mag: 435817, sign: true }, '*scores[2] == -6.65');
    assert(*scores.data[3] == FP16x16 { mag: 435817, sign: false }, '*scores[3] == 6.65');
}

#[test]
#[available_gas(200000000000)]
fn test_linear_classifier_binary_logistic() {
    let (mut classifier, X) = linear_classifier_helper_binary(POST_TRANSFORM::LOGISTIC);

    let (labels, mut scores) = LinearClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 1, 'labels[0]');
    assert(*labels[1] == 1, 'labels[1]');
    assert(labels.len() == 2, 'len(labels)');

    // ASSERT SCORES
    assert(*scores.data[0] == FP16x16 { mag: 4, sign: false }, '*scores[0] == 7.263436e-05');
    assert(*scores.data[1] == FP16x16 { mag: 65532, sign: false }, '*scores[1] == 9.999274e-01');
    assert(*scores.data[2] == FP16x16 { mag: 84, sign: false }, '*scores[2] == 1.292350e-03');
    assert(*scores.data[3] == FP16x16 { mag: 65452, sign: false }, '*scores[3] == 9.999983e-01');
}

#[test]
#[available_gas(200000000000)]
fn test_linear_classifier_binary_softmax() {
    let (mut classifier, X) = linear_classifier_helper_binary(POST_TRANSFORM::SOFTMAX);

    let (labels, mut scores) = LinearClassifierTrait::predict(ref classifier, X);
    // ASSERT LABELS
    assert(*labels[0] == 1, 'labels[0]');
    assert(*labels[1] == 1, 'labels[1]');
    assert(labels.len() == 2, 'len(labels)');

    // ASSERT SCORES
    assert(*scores.data[0] == FP16x16 { mag: 0, sign: false }, '*scores[0] == 5.276517e-09');
    assert(*scores.data[1] == FP16x16 { mag: 65535, sign: false }, '*scores[1] == 1.000000');
    assert(*scores.data[2] == FP16x16 { mag: 0, sign: false }, '*scores[2] == 1.674492e-06');
    assert(*scores.data[3] == FP16x16 { mag: 65535, sign: false }, '*scores[3] ==  9.999983e-01');
}

#[test]
#[available_gas(200000000000)]
fn test_linear_classifier_binary_softmax_zero() {
    let (mut classifier, X) = linear_classifier_helper_binary(POST_TRANSFORM::SOFTMAXZERO);

    let (labels, mut scores) = LinearClassifierTrait::predict(ref classifier, X);
    // ASSERT LABELS
    assert(*labels[0] == 1, 'labels[0]');
    assert(*labels[1] == 1, 'labels[1]');
    assert(labels.len() == 2, 'len(labels)');

    // ASSERT SCORES
    assert(*scores.data[0] == FP16x16 { mag: 0, sign: false }, '*scores[0] == 5.276517e-09');
    assert(*scores.data[1] == FP16x16 { mag: 65535, sign: false }, '*scores[1] == 1.000000');
    assert(*scores.data[2] == FP16x16 { mag: 0, sign: false }, '*scores[2] == 1.674492e-06');
    assert(*scores.data[3] == FP16x16 { mag: 65535, sign: false }, '*scores[3] ==  9.999983e-01');
}

#[test]
#[available_gas(200000000000)]
fn test_linear_classifier_unary_none() {
    let (mut classifier, X) = linear_classifier_helper_unary(POST_TRANSFORM::NONE);

    let (labels, mut scores) = LinearClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 1, 'labels[0]');
    assert(*labels[1] == 0, 'labels[1]');
    assert(labels.len() == 2, 'len(labels)');

    // ASSERT SCORES
    assert(*scores.data[0] == FP16x16 { mag: 146146, sign: false }, '*scores[0] == 2.23');
    assert(*scores.data[1] == FP16x16 { mag: 42596, sign: true }, '*scores[1] == -0.65');
}

#[test]
#[available_gas(200000000000)]
fn test_linear_classifier_unary_logistic() {
    let (mut classifier, X) = linear_classifier_helper_unary(POST_TRANSFORM::LOGISTIC);

    let (labels, mut scores) = LinearClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 1, 'labels[0]');
    assert(*labels[1] == 0, 'labels[1]');
    assert(labels.len() == 2, 'len(labels)');

    // ASSERT SCORES
    assert(*scores.data[0] == FP16x16 { mag: 59173, sign: false }, '*scores[0] == 0.902911');
    assert(*scores.data[1] == FP16x16 { mag: 22479, sign: false }, '*scores[1] == 0.34299');
}

#[test]
#[available_gas(200000000000)]
fn test_linear_classifier_unary_softmax() {
    let (mut classifier, X) = linear_classifier_helper_unary(POST_TRANSFORM::SOFTMAX);

    let (labels, mut scores) = LinearClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 1, 'labels[0]');
    assert(*labels[1] == 1, 'labels[1]');
    assert(labels.len() == 2, 'len(labels)');

    // ASSERT SCORES
    assert(*scores.data[0] == FP16x16 { mag: 65536, sign: false }, '*scores[0] == 1');
    assert(*scores.data[1] == FP16x16 { mag: 65536, sign: false }, '*scores[1] == 1');
}

#[test]
#[available_gas(200000000000)]
fn test_linear_classifier_unary_softmax_zero() {
    let (mut classifier, X) = linear_classifier_helper_unary(POST_TRANSFORM::SOFTMAXZERO);

    let (labels, mut scores) = LinearClassifierTrait::predict(ref classifier, X);

    // ASSERT LABELS
    assert(*labels[0] == 1, 'labels[0]');
    assert(*labels[1] == 1, 'labels[1]');
    assert(labels.len() == 2, 'len(labels)');

    // ASSERT SCORES
    assert(*scores.data[0] == FP16x16 { mag: 65536, sign: false }, '*scores[0] == 1');
    assert(*scores.data[1] == FP16x16 { mag: 65536, sign: false }, '*scores[1] == 1');
}


// ============ HELPER ============ //

fn linear_classifier_helper(
    post_transform: POST_TRANSFORM
) -> (LinearClassifier<FP16x16>, Tensor<FP16x16>) {
    let classlabels: Span<usize> = array![0, 1, 2].span();
    let classlabels = Option::Some(classlabels);

    let coefficients: Span<FP16x16> = array![
        FP16x16 { mag: 38011, sign: true },
        FP16x16 { mag: 19005, sign: true },
        FP16x16 { mag: 5898, sign: true },
        FP16x16 { mag: 38011, sign: false },
        FP16x16 { mag: 19005, sign: false },
        FP16x16 { mag: 5898, sign: false },
    ]
        .span();

    let intercepts: Span<FP16x16> = array![
        FP16x16 { mag: 176947, sign: false },
        FP16x16 { mag: 176947, sign: true },
        FP16x16 { mag: 32768, sign: false },
    ]
        .span();
    let intercepts = Option::Some(intercepts);

    let multi_class: usize = 0;

    let mut classifier: LinearClassifier<FP16x16> = LinearClassifier {
        classlabels, coefficients, intercepts, multi_class, post_transform
    };

    let mut X: Tensor<FP16x16> = TensorTrait::new(
        array![3, 2].span(),
        array![
            FP16x16 { mag: 0, sign: false },
            FP16x16 { mag: 65536, sign: false },
            FP16x16 { mag: 131072, sign: false },
            FP16x16 { mag: 196608, sign: false },
            FP16x16 { mag: 262144, sign: false },
            FP16x16 { mag: 327680, sign: false },
        ]
            .span()
    );

    (classifier, X)
}


fn linear_classifier_helper_binary(
    post_transform: POST_TRANSFORM
) -> (LinearClassifier<FP16x16>, Tensor<FP16x16>) {
    let classlabels: Span<usize> = array![0, 1].span();
    let classlabels = Option::Some(classlabels);

    let coefficients: Span<FP16x16> = array![
        FP16x16 { mag: 38011, sign: true },
        FP16x16 { mag: 19005, sign: true },
        FP16x16 { mag: 5898, sign: true },
    ]
        .span();

    let intercepts: Span<FP16x16> = array![FP16x16 { mag: 655360, sign: false },].span();
    let intercepts = Option::Some(intercepts);

    let multi_class: usize = 0;

    let mut classifier: LinearClassifier<FP16x16> = LinearClassifier {
        classlabels, coefficients, intercepts, multi_class, post_transform
    };

    let mut X: Tensor<FP16x16> = TensorTrait::new(
        array![2, 3].span(),
        array![
            FP16x16 { mag: 0, sign: false },
            FP16x16 { mag: 65536, sign: false },
            FP16x16 { mag: 131072, sign: false },
            FP16x16 { mag: 196608, sign: false },
            FP16x16 { mag: 262144, sign: false },
            FP16x16 { mag: 327680, sign: false },
        ]
            .span()
    );

    (classifier, X)
}

fn linear_classifier_helper_unary(
    post_transform: POST_TRANSFORM
) -> (LinearClassifier<FP16x16>, Tensor<FP16x16>) {
    let classlabels: Span<usize> = array![1].span();
    let classlabels = Option::Some(classlabels);

    let coefficients: Span<FP16x16> = array![
        FP16x16 { mag: 38011, sign: true },
        FP16x16 { mag: 19005, sign: true },
        FP16x16 { mag: 5898, sign: true },
    ]
        .span();

    let intercepts: Span<FP16x16> = array![FP16x16 { mag: 176947, sign: false },].span();
    let intercepts = Option::Some(intercepts);

    let multi_class: usize = 0;

    let mut classifier: LinearClassifier<FP16x16> = LinearClassifier {
        classlabels, coefficients, intercepts, multi_class, post_transform
    };

    let mut X: Tensor<FP16x16> = TensorTrait::new(
        array![2, 3].span(),
        array![
            FP16x16 { mag: 0, sign: false },
            FP16x16 { mag: 65536, sign: false },
            FP16x16 { mag: 131072, sign: false },
            FP16x16 { mag: 196608, sign: false },
            FP16x16 { mag: 262144, sign: false },
            FP16x16 { mag: 327680, sign: false },
        ]
            .span()
    );

    (classifier, X)
}
