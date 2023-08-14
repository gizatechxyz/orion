#tensor.sqrt

```rust
    fn sqrt(self: @Tensor<T>) -> Tensor<T>;
```

Computes the square root of all elements of the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.


## Returns

A new `Tensor<T>` of the same shape as the input tensor with 
the arctangent (inverse of tangent) value of all elements in the input tensor.

## Example

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};
use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;

fn sqrt_example() -> Tensor<FixedType> {
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    let tensor = TensorTrait::<FixedType>::new(
        shape: array![3].span(),
        data: array![
            FixedTrait::new_unscaled(0, false),
            FixedTrait::new_unscaled(1, false),
            FixedTrait::new_unscaled(2, false),
        ].span(),
        extra: Option::Some(extra)
    );

    return tensor.sqrt();
}
>>> [0,8388608,11863169]
// The fixed point representation of
// [0,1,1.4142...]
```
   