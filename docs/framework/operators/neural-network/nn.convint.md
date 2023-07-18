# NNTrait::convint

```rust
fn convint(inputs: Tensor<T>, weights: Tensor<T>, bias: Tensor<T>, kernel_size: Option<usize>, strides: Option<usize>) -> Tensor<T>;
```

Performs a convolution of the input tensor using the provided weights and bias.

## Args

* `inputs`(`Tensor<T>`) - A tensor larger than or equal to 3D representing the input tensor.
* `weights`(`Tensor<T>`) - A tensor representing the weights.
* `bias`(`Tensor<T>`) - A tensor representing the bias.
* `kernel_size` (`Option<usize>`) - The size of the convolution kernel. If not present, it is inferred from input 'w'.
* `strides` (`Option<usize>`) - The stides along the axes. If not present, it defaults to 1.

## Panics

* This function asserts that the input tensor `inputs` must be at least 3D.

## Returns

A `Tensor<T>` representing the result of the convolution.

## Examples

```rust
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_i32::NN_i32;

fn convint_layer_example() -> Tensor<u32> {
    // We instantiate inputs here.
    // inputs = [-71, 38, 62]
    let inputs = i32_inputs_helper();

    // We instantiate weights here.
    // weights = [[-8, 64, 40], [-33, -34, -20]]
    let weights = i32_weights_helper();

    // We instantiate bias here.
    // bias = [61, -71]
    let bias = i32_bias_helper();

    // We can call `linear` function as follows.
    return NNTrait::convint(inputs, weights, bias, Option::Some(3), Option::None());
}
>>> [5541, -260]
````
