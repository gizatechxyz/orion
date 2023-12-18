# LinearRegressorTrait::predict

```rust 
   fn predict(ref self: LinearRegressor<T>, X: Tensor<T>) -> Tensor<T>;
```

Linear Regressor. Performs the generalized linear regression evaluation.

## Args

* `self`: LinearRegressor<T> - A LinearRegressor object.
* `X`:  Input 2D tensor.

## Returns

* Tensor<T> containing the generalized linear regression evaluation of the input X.

## Type Constraints

`LinearRegressor` and `X` must be fixed points

## Examples

```rust
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, U32Tensor, FP16x16TensorAdd};
use orion::operators::ml::linear::linear_regressor::{
    LinearRegressorTrait, POST_TRANSFORM, LinearRegressor
};
use orion::numbers::{FP16x16, FixedTrait};
use orion::operators::nn::{NNTrait, FP16x16NN};

fn example_linear_regressor() -> Tensor<FP16x16> {

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

    ]
        .span();

    let intercepts: Span<FP16x16> = array![
        FP16x16 { mag: 32768, sign: false },

    ]
        .span();
    let intercepts = Option::Some(intercepts);    

    let target : usize = 1;
    let post_transform = POST_TRANSFORM::NONE;

    let mut regressor: LinearRegressor<FP16x16> = LinearRegressor {
        coefficients,
        intercepts,
        target,
        post_transform
    };

    let scores = LinearRegressorTrait::predict(ref regressor, X);

    scores
}

>>> 
[[-0.27], [-1.21], [-2.15]]



fn example_linear_regressor_2() -> Tensor<FP16x16> {

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
        FP16x16 { mag: 32768, sign: false },
        FP16x16 { mag: 45875, sign: false },

    ]
        .span();
    let intercepts = Option::Some(intercepts);  

    let target = 2;
    let post_transform = POST_TRANSFORM::NONE;

    let mut regressor: LinearRegressor<FP16x16> = LinearRegressor {
        coefficients,
        intercepts,
        target,
        post_transform
    };

    let scores = LinearRegressorTrait::predict(ref regressor, X);

    scores
}

>>>
[[-0.27, -0.07], [-1.21, -1.01], [-2.15, -1.95]]   
```

