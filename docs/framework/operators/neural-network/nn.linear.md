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
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_i32::NN_i32;

fn linear_layer_example() -> Tensor<u32> {
    // We instantiate inputs here.
    // inputs = [-71, 38, 62]
    let inputs = i32_inputs_helper();

    // We instantiate weights here.
    // weights = [[-8, 64, 40], [-33, -34, -20]]
    let weights = i32_weights_helper();

    // We instantiate bias here.
    // weights = [61, -71]
    let weights = u32_bias_helper();

    // We can call `linear` function as follows.
    return NNTrait::linear(inputs, weights, bias);
}
>>> [5541, -260]
````
