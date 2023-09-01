#tensor.atan

```rust
    fn atan(self: @Tensor<T>) -> Tensor<T>;
```

Computes the arctangent (inverse of tangent) of all elements of the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.


## Returns

A new `Tensor<T>` of the same shape as the input tensor with 
the arctangent (inverse of tangent) value of all elements in the input tensor.

## Example

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, Tensor_fp8x23};
use orion::numbers::{FixedTrait, FP8x23};

fn atan_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<FP8x23>::new(
        shape: array![3].span(),
        data: array![
            FixedTrait::new_unscaled(0, false),
            FixedTrait::new_unscaled(1, false),
            FixedTrait::new_unscaled(2, false),
        ]
            .span(),
    );

    return tensor.atan();
}
>>> [0,6588397,9287028]
// The fixed point representation of
// [0,0.7853...,1.1071...]
```
   