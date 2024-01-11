use orion::numbers::FP16x16;
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, U32Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP16x16TensorPartialEq;

use orion::operators::ml::svm::svm_regressor::{SVMRegressorTrait, POST_TRANSFORM, SVMRegressor};
use orion::operators::ml::svm::core::{KERNEL_TYPE};


#[test]
#[available_gas(200000000000)]
fn test_svm_regressor_linear() {
    let kernel_params: Span<FP16x16> = array![
        FP16x16 { mag: 27812, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 196608, sign: false }
    ]
        .span();
    let kernel_type = KERNEL_TYPE::LINEAR;
    let (mut regressor, X) = svm_regressor_helper(kernel_type, kernel_params);

    let scores = SVMRegressorTrait::predict(ref regressor, X);

    let mut expected_scores: Tensor<FP16x16> = TensorTrait::new(
        array![3, 1].span(),
        array![
            FP16x16 { mag: 30684, sign: true },
            FP16x16 { mag: 14908, sign: false },
            FP16x16 { mag: 60501, sign: false },
        ]
            .span()
    );
    assert_eq(scores, expected_scores);
}

#[test]
#[available_gas(200000000000)]
fn test_svm_regressor_poly() {
    let kernel_params: Span<FP16x16> = array![
        FP16x16 { mag: 22456, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 196608, sign: false }
    ]
        .span();

    let kernel_type = KERNEL_TYPE::POLY;
    let (mut regressor, X) = svm_regressor_helper(kernel_type, kernel_params);

    let scores = SVMRegressorTrait::predict(ref regressor, X);

    let mut expected_scores: Tensor<FP16x16> = TensorTrait::new(
        array![3, 1].span(),
        array![
            FP16x16 { mag: 34542, sign: false },
            FP16x16 { mag: 35623, sign: false },
            FP16x16 { mag: 35815, sign: false },
        ]
            .span()
    );
    assert_eq(scores, expected_scores);
}


#[test]
#[available_gas(200000000000)]
fn test_svm_regressor_rbf() {
    let kernel_params: Span<FP16x16> = array![
        FP16x16 { mag: 19848, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 196608, sign: false }
    ]
        .span();
    let kernel_type = KERNEL_TYPE::RBF;
    let (mut regressor, X) = svm_regressor_helper(kernel_type, kernel_params);

    let scores = SVMRegressorTrait::predict(ref regressor, X);

    let mut expected_scores: Tensor<FP16x16> = TensorTrait::new(
        array![3, 1].span(),
        array![
            FP16x16 { mag: 19376, sign: false },
            FP16x16 { mag: 31318, sign: false },
            FP16x16 { mag: 45566, sign: false },
        ]
            .span()
    );
    assert_eq(scores, expected_scores);
}

#[test]
#[available_gas(200000000000)]
fn test_svm_regressor_sigmoid() {
    let kernel_params: Span<FP16x16> = array![
        FP16x16 { mag: 20108, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 196608, sign: false }
    ]
        .span();
    let kernel_type = KERNEL_TYPE::SIGMOID;
    let (mut regressor, X) = svm_regressor_helper(kernel_type, kernel_params);

    let scores = SVMRegressorTrait::predict(ref regressor, X);

    let mut expected_scores: Tensor<FP16x16> = TensorTrait::new(
        array![3, 1].span(),
        array![
            FP16x16 { mag: 15683, sign: false },
            FP16x16 { mag: 29421, sign: false },
            FP16x16 { mag: 43364, sign: false },
        ]
            .span()
    );
    assert_eq(scores, expected_scores);
}

#[test]
#[available_gas(200000000000)]
fn test_svm_regressor_linear_one_class_0() {
    let post_transform = POST_TRANSFORM::NONE;
    let one_class = 0;
    let (mut regressor, X) = svm_regressor_linear_helper(post_transform, one_class);

    let scores = SVMRegressorTrait::predict(ref regressor, X);

    let mut expected_scores: Tensor<FP16x16> = TensorTrait::new(
        array![3, 1].span(),
        array![
            FP16x16 { mag: 63484, sign: false },
            FP16x16 { mag: 74218, sign: false },
            FP16x16 { mag: 84953, sign: false },
        ]
            .span()
    );
    assert_eq(scores, expected_scores);
}

#[test]
#[available_gas(200000000000)]
fn test_svm_regressor_linear_one_class_1() {
    let post_transform = POST_TRANSFORM::NONE;
    let one_class = 1;
    let (mut regressor, X) = svm_regressor_linear_helper(post_transform, one_class);

    let scores = SVMRegressorTrait::predict(ref regressor, X);

    let mut expected_scores: Tensor<FP16x16> = TensorTrait::new(
        array![3, 1].span(),
        array![
            FP16x16 { mag: 65536, sign: false },
            FP16x16 { mag: 65536, sign: false },
            FP16x16 { mag: 65536, sign: false },
        ]
            .span()
    );
    assert_eq(scores, expected_scores);
}


// ============ HELPER ============ //

fn svm_regressor_helper(
    kernel_type: KERNEL_TYPE, kernel_params: Span<FP16x16>
) -> (SVMRegressor<FP16x16>, Tensor<FP16x16>) {
    let coefficients: Span<FP16x16> = array![
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 65536, sign: true },
        FP16x16 { mag: 54959, sign: false },
        FP16x16 { mag: 54959, sign: true },
        FP16x16 { mag: 29299, sign: false },
        FP16x16 { mag: 65536, sign: true },
        FP16x16 { mag: 36236, sign: false }
    ]
        .span();

    let n_supports: usize = 7;
    let one_class: usize = 0;
    let rho: Span<FP16x16> = array![FP16x16 { mag: 35788, sign: false }].span();

    let support_vectors: Span<FP16x16> = array![
        FP16x16 { mag: 8421, sign: true },
        FP16x16 { mag: 5842, sign: false },
        FP16x16 { mag: 4510, sign: false },
        FP16x16 { mag: 5202, sign: true },
        FP16x16 { mag: 14783, sign: true },
        FP16x16 { mag: 17380, sign: true },
        FP16x16 { mag: 60595, sign: false },
        FP16x16 { mag: 1674, sign: true },
        FP16x16 { mag: 38669, sign: true },
        FP16x16 { mag: 63803, sign: false },
        FP16x16 { mag: 87720, sign: true },
        FP16x16 { mag: 22236, sign: false },
        FP16x16 { mag: 61816, sign: false },
        FP16x16 { mag: 34267, sign: true },
        FP16x16 { mag: 36418, sign: false },
        FP16x16 { mag: 27471, sign: false },
        FP16x16 { mag: 28421, sign: false },
        FP16x16 { mag: 69270, sign: true },
        FP16x16 { mag: 152819, sign: false },
        FP16x16 { mag: 4065, sign: false },
        FP16x16 { mag: 62274, sign: true }
    ]
        .span();

    let post_transform = POST_TRANSFORM::NONE;

    let mut regressor: SVMRegressor<FP16x16> = SVMRegressor {
        coefficients,
        kernel_params,
        kernel_type,
        n_supports,
        one_class,
        post_transform,
        rho,
        support_vectors,
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

fn svm_regressor_linear_helper(
    post_transform: POST_TRANSFORM, one_class: usize
) -> (SVMRegressor<FP16x16>, Tensor<FP16x16>) {
    let coefficients: Span<FP16x16> = array![
        FP16x16 { mag: 18540, sign: false },
        FP16x16 { mag: 1746, sign: true },
        FP16x16 { mag: 1097, sign: false }
    ]
        .span();
    let kernel_params: Span<FP16x16> = array![
        FP16x16 { mag: 65, sign: false },
        FP16x16 { mag: 0, sign: false },
        FP16x16 { mag: 196608, sign: false }
    ]
        .span();
    let kernel_type = KERNEL_TYPE::LINEAR;
    let n_supports: usize = 0;
    let rho: Span<FP16x16> = array![FP16x16 { mag: 81285, sign: false }].span();

    let support_vectors: Span<FP16x16> = array![].span();

    let mut regressor: SVMRegressor<FP16x16> = SVMRegressor {
        coefficients,
        kernel_params,
        kernel_type,
        n_supports,
        one_class,
        post_transform,
        rho,
        support_vectors,
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

    (regressor, X)
}
