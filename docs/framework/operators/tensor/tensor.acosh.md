# tensor.acosh

```rust 
    fn acosh(self: @Tensor<T>) -> Tensor<FixedType>;
```

Computes the inverse hyperbolic cosine of all elements of the input tensor.
$$
y_i=acosh({x_i})
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `FixedType` with the hyperblic cosine of the elements of the input tensor.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};
use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;

fn acosh_example() -> Tensor<FixedType> {
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    let tensor = TensorTrait::<FixedType>::new(
        shape: array![2,2].span(),
        data: array![
            FixedTrait::new_unscaled(1, false),
            FixedTrait::new_unscaled(2, false),
            FixedTrait::new_unscaled(3, false),
            FixedTrait::new_unscaled(4, false)
        ].span(),
        extra: Option::Some(extra)
    );

   return tensor.acosh();
}
>>> [[0,11047444],[14786996,17309365]]
// The fixed point representation of
// [[0, 1.31696],[1.76275, 2.06344]]
```
