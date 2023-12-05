#tensor.not

```rust
    fn not(self: @Tensor<bool>) -> Tensor<bool;
```

Computes the negation of the elements in the bool type input tensor.

## Args

* `self`(`@Tensor<bool>`) - The input tensor.


## Returns

A new `Tensor<bool>` of the same shape as the input tensor with 
the negation of all elements in the input tensor.

## Example

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, BoolTensor};
use orion::numbers::{i32, IntegerTrait};

fn not_example() -> Tensor<bool> {
    let tensor = TensorTrait::new(
        shape: array![3].span(),
        data: array![
            true, true, false
        ]
            .span(),
    );

    return tensor.not();
}
>>> [true, true, false]
```
