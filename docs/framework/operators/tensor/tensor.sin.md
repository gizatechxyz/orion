#tensor.sin

```rust
    fn sin(self: @Tensor<T>) -> Tensor<T>;
```

Computes the sine of all elements of the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.


## Returns

A new `Tensor<T>` of the same shape as the input tensor with 
the sine value of all elements in the input tensor.

## Type Constraints

Constrain input and output types to fixed point tensors.

## Example

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
use orion::numbers::{FP8x23, FixedTrait};

fn sin_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::<FP8x23>::new(
        shape: array![3].span(),
        data: array![
            FixedTrait::new_unscaled(0, false),
            FixedTrait::new_unscaled(1, false),
            FixedTrait::new_unscaled(2, false)
        ]
            .span(),
    );

    return tensor.sin();
}
>>> [0,7058770,7627740]
// The fixed point representation of
// [0,0.8414...,0.9092...]
```
