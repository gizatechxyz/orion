# tensor.asinh

```rust 
    fn asinh(self: @Tensor<T>) -> Tensor<FixedType>;
```

Computes the inverse hyperbolic sine of all elements of the input tensor.
$$
y_i=asinh({x_i})
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `FixedType` with the hyperblic sine of the elements of the input tensor.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};
use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;

fn asinh_example() -> Tensor<FixedType> {
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

   return tensor.asinh();
}
>>> [[0,7393498],[12110093,15254235]]
// The fixed point representation of
// [[0, 0.8814],[1.44364, 1.8185]]
```
