# LinearClassifierTrait::predict

```rust 
   fn predict(ref self: LinearClassifier<T>, X: Tensor<T>) -> Tensor<T>;
```

Linear Regressor. Performs the linear classification.

## Args

* `self`: LinearClassifier<T> - A LinearClassifier object.
* `X`:  Input 2D tensor.

## Returns

* Tensor<T> containing the generalized linear regression evaluation of the input X.

## Type Constraints

`LinearClassifier` and `X` must be fixed points

## Examples

```rust
use orion::numbers::FP16x16;
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor, U32Tensor};

use orion::operators::ml::linear::linear_classifier::{
    LinearClassifierTrait, POST_TRANSFORM, LinearClassifier
};

fn linear_classifier_helper(
    post_transform: POST_TRANSFORM
) -> (LinearClassifier<FP16x16>, Tensor<FP16x16>) {

    let classlabels: Span<usize> = array![0, 1, 2].span();
    let classlabels = Option::Some(classlabels);

    let classlabels_strings: Option<Span<FP16x16>> = Option::None;

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
        classlabels,
        coefficients,
        intercepts,
        multi_class,
        post_transform
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

fn linear_classifier_multi_softmax() -> (Span<usize>, Tensor<FP16x16>) {
    let (mut classifier, X) = linear_classifier_helper(POST_TRANSFORM::SOFTMAX);

    let (labels, mut scores) = LinearClassifierTrait::predict(ref classifier, X);

    (labels, scores)
}

>>> 
([0, 2, 2],
 [
    [0.852656, 0.009192, 0.138152],
    [0.318722, 0.05216, 0.629118],
    [0.036323, 0.090237, 0.87344]
 ])
```