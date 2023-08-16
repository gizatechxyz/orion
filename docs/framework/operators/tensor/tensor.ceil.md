#tensor.ceil

```rust
    fn ceil(self: @Tensor<T>) -> Tensor<T>;
```

Rounds up the value of each element in the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.


## Returns

A new `Tensor<T>` of the same shape as the input tensor with 
the rounded up value of all elements in the input tensor.

## Example

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_fp::{Tensor_fp};
use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;

fn ceil_example() -> Tensor<FixedType> {
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    let tensor = TensorTrait::<FixedType>::new(
        shape: array![3].span(),
        data: array![
            FixedTrait::new(29998, false), // 0.003576
            FixedTrait::new(100663252, false), // 11.9999947548
            FixedTrait::new(100663252, true) // -11.9999947548
        ].span(),
        extra: Option::Some(extra)
    );

    return tensor.ceil();
}
>>> [1,12,-11]
```
