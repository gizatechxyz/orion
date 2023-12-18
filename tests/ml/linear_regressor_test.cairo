use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, U32Tensor, FP16x16TensorAdd};
use orion::operators::ml::linear::linear_regressor::{
    LinearRegressorTrait, POST_TRANSFORM, LinearRegressor
};
use orion::numbers::{FP16x16, FixedTrait};

use core::debug::PrintTrait;

use orion::operators::nn::{NNTrait, FP16x16NN};


#[test]
#[available_gas(200000000000)]
fn test_linear_regressor() {
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

    let coefficients: Span<FP16x16> = array![
        FP16x16 { mag: 19661, sign: false }, FP16x16 { mag: 50463, sign: true },
    ]
        .span();

    let intercepts: Span<FP16x16> = array![FP16x16 { mag: 32768, sign: false },].span();
    let intercepts = Option::Some(intercepts);

    let target: usize = 1;
    let post_transform = POST_TRANSFORM::NONE;

    let mut regressor: LinearRegressor<FP16x16> = LinearRegressor {
        coefficients, intercepts, target, post_transform
    };

    let scores = LinearRegressorTrait::predict(ref regressor, X);

    assert(*scores.data[0] == FP16x16 { mag: 17695, sign: true }, '*scores[0] == -0.27');
    assert(*scores.data[1] == FP16x16 { mag: 79299, sign: true }, '*scores[1] == -1.21');
    assert(*scores.data[2] == FP16x16 { mag: 140903, sign: true }, '*scores[2] == -2.15');
}

#[test]
#[available_gas(200000000000)]
fn test_linear_regressor_2() {
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

    let coefficients: Span<FP16x16> = array![
        FP16x16 { mag: 19661, sign: false },
        FP16x16 { mag: 50463, sign: true },
        FP16x16 { mag: 19661, sign: false },
        FP16x16 { mag: 50463, sign: true },
    ]
        .span();

    let intercepts: Span<FP16x16> = array![
        FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 45875, sign: false },
    ]
        .span();
    let intercepts = Option::Some(intercepts);

    let target = 2;
    let post_transform = POST_TRANSFORM::NONE;

    let mut regressor: LinearRegressor<FP16x16> = LinearRegressor {
        coefficients, intercepts, target, post_transform
    };

    let scores = LinearRegressorTrait::predict(ref regressor, X);

    assert(*scores.data[0] == FP16x16 { mag: 17695, sign: true }, '*scores[0] == -0.27');
    assert(*scores.data[1] == FP16x16 { mag: 4588, sign: true }, '*scores[1] == -0.07');
    assert(*scores.data[2] == FP16x16 { mag: 79299, sign: true }, '*scores[2] == -1.21');
    assert(*scores.data[3] == FP16x16 { mag: 66192, sign: true }, '*scores[3] == -1.01');
    assert(*scores.data[4] == FP16x16 { mag: 140903, sign: true }, '*scores[4] == -2.15');
    assert(*scores.data[5] == FP16x16 { mag: 127796, sign: true }, '*scores[5] == -1.95');
}
