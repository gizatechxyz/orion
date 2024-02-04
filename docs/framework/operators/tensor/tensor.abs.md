#tensor.abs

```rust
    fn abs(self: @Tensor<T>) -> Tensor<T>;
```

Computes the absolute value of all elements in the input tensor.

## Args

* `self`(`@Tensor<T>`) - The input tensor.


## Returns

A new `Tensor<T>` of the same shape as the input tensor with 
the absolute value of all elements in the input tensor.

## Example

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor};

fn abs_example() -> Tensor<i32> {
    let tensor = TensorTrait::new(
        shape: array![3].span(),
        data: array![
            -1, -2, 3
        ]
            .span(),
    );

    return tensor.abs();
}
>>> [1, 2, 3]
```
