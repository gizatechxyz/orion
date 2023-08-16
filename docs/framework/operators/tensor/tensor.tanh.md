# tensor.tanh

```rust 
    fn tanh(self: @Tensor<T>) -> Tensor<FixedType>;
```

Computes the hyperbolic tangent of all elements of the input tensor.
$$
y_i=tanh({x_i})
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `FixedType` with the hyperbolic tangent of the elements of the input tensor.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};
use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;

fn tanh_example() -> Tensor<FixedType> {
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    let tensor = TensorTrait::<FixedType>::new(
        shape: array![2,2].span(),
        data: array![
            FixedTrait::new_unscaled(0, false),
            FixedTrait::new_unscaled(1, false),
            FixedTrait::new_unscaled(2, false),
            FixedTrait::new_unscaled(3, false)
        ].span(),
        extra: Option::Some(extra)
    );

   return tensor.tanh();
}
>>> [[0,6388715],[8086850,8347125]]
// The fixed point representation of
// [[0, 0.761594],[0.96403, 0.9951]]
```
