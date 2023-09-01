# NNTrait::leaky_relu

```rust
 fn leaky_relu(inputs: @Tensor<T>, alpha: @F) -> Tensor<F>
```

Applies the leaky rectified linear unit (Leaky ReLU) activation function element-wise to a given tensor.

The Leaky ReLU function is defined as f(x) = alpha * x if x < 0, f(x) = x otherwise, where x is the input element.

## Args
* `inputs`(`@Tensor<T>`) - A snapshot of a tensor to which the Leaky ReLU function will be applied.
* `alpha`(`@F`) - A snapshot of a fixed point scalar that defines the alpha value of the Leaky ReLU function.

## Returns
A new fixed point tensor with the same shape as the input tensor and the Leaky ReLU function applied element-wise.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, Tensor_i32_fp8x23};
use orion::operators::nn::{NNTrait, NN_i32_fp8x23};
use orion::numbers::{i32, FP8x23, IntegerTrait, FixedTrait};

fn leaky_relu_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<i32>::new(
        shape: array![2, 3].span(),
        data: array![
            IntegerTrait::new(1, false),
            IntegerTrait::new(2, false),
            IntegerTrait::new(1, true),
            IntegerTrait::new(2, true),
            IntegerTrait::new(0, false),
            IntegerTrait::new(0, false),
        ]
            .span(),
    );
    let alpha = FixedTrait::from_felt(838861); // 0.1

    return NNTrait::leaky_relu(@tensor, @alpha);
}
>>> [[8388608, 16777216, 838861], [1677722, 0, 0]]
     // The fixed point representation of
    [[1, 2, 0.1], [0.2, 0, 0]]
```
