# tensor.sinh

```rust 
    fn sinh(self: @Tensor<T>) -> Tensor<FixedType>;
```

Computes the hyperbolic sine of all elements of the input tensor.
$$
y_i=sinh({x_i})
$$

## Args

* `self`(`@Tensor<T>`) - The input tensor.

## Returns

Returns a new tensor in `FixedType` with the hyperbolic sine of the elements of the input tensor.

## Examples

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};
use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;

fn sinh_example() -> Tensor<FixedType> {
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

   return tensor.sinh();
}
>>> [[0,9858303],[30424311,84036026]]
// The fixed point representation of
// [[0, 1.175201],[3.62686, 10.0178749]]
```
