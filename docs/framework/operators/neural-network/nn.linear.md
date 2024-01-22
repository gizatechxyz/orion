# NNTrait::linear

```rust
fn linear(inputs: Tensor<T>, weights: Tensor<T>, bias: Tensor<T>) -> Tensor<T>
```

Performs a linear transformation of the input tensor using the provided weights and bias.

## Args

* `tensor`(`@Tensor<T>`) - A 1D tensor representing the input tensor.
* `weights`(`@Tensor<T>`) - A 2D tensor representing the weights.
* `bias`(`@Tensor<T>`) - A 1D tensor representing the bias.

## Panics

* This function asserts that the input tensor `inputs` must be 1D, weights tensor must be 2D, and bias tensor must be 1D.

## Returns

A `Tensor<T>` representing the result of the linear transformation.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor};
use orion::operators::nn::{NNTrait, I32NN};

fn linear_example() -> Tensor<i32> {
    // We instantiate inputs here.
    let inputs = TensorTrait::<i32>::new(
        shape: array![3].span(),
        data: array![
            -71, 38, 62,
        ]
            .span(),
    );

    // We instantiate weights here.
    let weights = TensorTrait::<i32>::new(
        shape: array![2, 3].span(),
        data: array![
            -8,
            64,
            40,
            -33,
            -34,
            -20,
        ]
            .span(),
    );

    // We instantiate bias here.
    let bias = TensorTrait::<i32>::new(
        shape: array![2].span(),
        data: array![61, -61].span(),
    );

    return NNTrait::linear(inputs, weights, bias);
}
>>> [5541, -250]
````
