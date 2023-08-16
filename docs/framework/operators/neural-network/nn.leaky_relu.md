# NNTrait::leaky_relu

```rust
 fn leaky_relu(inputs: @Tensor<T>, alpha: @FixedType) -> Tensor<FixedType>
```

Applies the leaky rectified linear unit (Leaky ReLU) activation function element-wise to a given tensor.

The Leaky ReLU function is defined as f(x) = alpha * x if x < 0, f(x) = x otherwise, where x is the input element.

## Args
* `inputs`(`@Tensor<T>`) - A snapshot of a tensor to which the Leaky ReLU function will be applied.
* `alpha`(`@FixedType`) - A snapshot of a FixedType scalar that defines the alpha value of the Leaky ReLU function.

## Returns
A new FixedType tensor with the same shape as the input tensor and the Leaky ReLU function applied element-wise.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32::{Tensor_i32};
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
use orion::numbers::signed_integer::i32::{i32, IntegerTrait};
use orion::numbers::fixed_point::core::{FixedImpl, FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;

fn leaky_relu_example() -> Tensor<FixedType> {
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
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
        extra: Option::Some(extra)
    );
    let alpha = FixedTrait::from_felt(838861); // 0.1

    return NNTrait::leaky_relu(@tensor, @alpha);
}
>>> [[8388608, 16777216, 838861], [1677722, 0, 0]]
     // The fixed point representation of
    [[1, 2, 0.1], [0.2, 0, 0]]
```
