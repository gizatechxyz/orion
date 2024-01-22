# tensor.identity

```rust 
   fn identity(self: @Tensor<T>) -> Tensor<T>;
```

Return a Tensor with the same shape and contents as input.

## Args

* `self`(`@Tensor<T>`) - Input tensor.

## Returns 

A new `Tensor<T>` to copy input into.

## Example

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, i32Tensor};

fn identity_example() -> Tensor<i32> {
    let tensor = TensorTrait::<i32>::new(
        shape: array![2, 2].span(), 
        data: array![1, 2, 3, 4].span(), 
    );
    let t_identity = tensor.identity();
    t_identity
}
>>> [[1 2] [3 4]] // A Tensor with the same shape and contents as input
```
