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

## Type Constraints

Constrain input and output types to fixed point tensors.

## Example

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, FP8x23Tensor};
use orion::numbers::{FP8x23, FixedTrait};

fn ceil_example() -> Tensor<FP8x23> {
    let tensor = TensorTrait::new(
        shape: array![3].span(),
        data: array![
            FixedTrait::new(29998, false), // 0.003576
            FixedTrait::new(100663252, false), // 11.9999947548
            FixedTrait::new(100663252, true) // -11.9999947548
        ]
            .span(),
    );

    return tensor.ceil();
}
>>> [1,12,-11]
```
