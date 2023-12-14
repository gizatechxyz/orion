# NNTrait::thresholded_relu

```rust
 fn thresholded_relu(tensor: @Tensor<T>, alpha: @T) -> Tensor<T>
```

Applies the thresholded rectified linear unit (Thresholded ReLU) activation function element-wise to a given tensor.

The Thresholded ReLU function is defined as f(x) = x if x > alpha, f(x) = 0 otherwise, where x is the input element.

## Args
* `tensor`(`@Tensor<T>`) - A snapshot of a tensor to which the Leaky ReLU function will be applied.
* `alpha`(`@T`) - A snapshot of a fixed point scalar that defines the alpha value of the Thresholded ReLU function.

## Returns
A new fixed point tensor with the same shape as the input tensor and the Thresholded ReLU function applied element-wise.

## Type Constraints

Constrain input and output types to fixed point tensors.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP8x23};
use orion::operators::nn::{NNTrait, FP8x23NN};
use orion::numbers::{FP8x23, FixedTrait};

fn thresholded_relu_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<FP8x23>::new(
        shape: array![2, 2].span(),
        data: array![
            FixedTrait::new(0, false),
            FixedTrait::new(256, false),
            FixedTrait::new(512, false),
            FixedTrait::new(257, false),
        ]
            .span(),
    );
    let alpha = FixedTrait::from_felt(256); // 1.0

    return NNTrait::leaky_relu(@tensor, @alpha);
}
>>> [[0, 0], [512, 257]]
```
