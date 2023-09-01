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
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, Tensor_i32_fp8x23};
use orion::operators::nn::{NNTrait, NN_i32_fp8x23};
use orion::numbers::{i32, IntegerTrait};

fn linear_example() -> Tensor<i32> {
    // We instantiate inputs here.
    let inputs = TensorTrait::<i32>::new(
        shape: array![3].span(),
        data: array![
            IntegerTrait::new(71, true), IntegerTrait::new(38, false), IntegerTrait::new(62, false),
        ]
            .span(),
    );

    // We instantiate weights here.
    let weights = TensorTrait::<i32>::new(
        shape: array![2, 3].span(),
        data: array![
            IntegerTrait::new(8, true),
            IntegerTrait::new(64, false),
            IntegerTrait::new(40, false),
            IntegerTrait::new(33, true),
            IntegerTrait::new(34, true),
            IntegerTrait::new(20, true),
        ]
            .span(),
    );

    // We instantiate bias here.
    let bias = TensorTrait::<i32>::new(
        shape: array![2].span(),
        data: array![IntegerTrait::new(61, false), IntegerTrait::new(61, true),].span(),
    );

    return NNTrait::linear(inputs, weights, bias);
}
>>> [5541, -250]
````
