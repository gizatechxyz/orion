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

## Type Constraints

Constrain input and output types to fixed point tensors.

## Example

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
use orion::numbers::{FixedTrait, FP8x23};

fn sqrt_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<FP8x23>::new(
        shape: array![3].span(),
        data: array![
            FixedTrait::new_unscaled(0, false),
            FixedTrait::new_unscaled(1, false),
            FixedTrait::new_unscaled(2, false),
        ]
            .span(),
    );

    return tensor.sqrt();
}
>>> [0,8388608,11863169]
// The fixed point representation of
// [0,1,1.4142...]
```
   