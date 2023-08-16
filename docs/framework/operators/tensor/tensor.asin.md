#tensor.asin

```rust
    fn asin(self: @Tensor<T>) -> Tensor<T>;
```

Computes the arcsine (inverse of sine) of all elements of the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.


## Returns

A new `Tensor<T>` of the same shape as the input tensor with 
the arcsine value of all elements in the input tensor.

## Example

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};
use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;

fn asin_example() -> Tensor<FixedType> {
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    let tensor = TensorTrait::<FixedType>::new(
        shape: array![2].span(),
        data: array![
            FixedTrait::new_unscaled(0, false),
            FixedTrait::new_unscaled(1, false),
        ].span(),
        extra: Option::Some(extra)
    );

   return tensor.asin();
}
>>> [0, 13176794]
// The fixed point representation of
// [0, 1.5707...]
```
