#tensor.where

```rust
    fn where(self: @Tensor<T>, x: @Tensor<T>, y: @Tensor<T>) -> Tensor<T>;
```

Computes a new tensor by selecting values from tensor x (resp. y) at
indices where the condition is 1 (resp. 0).

## Args

* `self`(`@Tensor<T>`) - The condition tensor
* `x`(`@Tensor<T>`) - The first input tensor
* `y`(`@Tensor<T>`) - The second input tensor

## Panics

* Panics if the shapes are not equal or broadcastable

## Returns

Return a new `Tensor<T>` of the same shape as the input with elements 
chosen from x or y depending on the condition.

## Example

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn where_example() -> Tensor<u32> {
    let tensor_cond = TensorTrait::<u32>::new(
        shape: array![2, 2].span(), data: array![0, 1, 0, 1].span(),
    );

    let tensor_x = TensorTrait::<u32>::new(
        shape: array![2, 2].span(), data: array![2, 4, 6, 8].span(),
    );

    let tensor_y = TensorTrait::<u32>::new(
        shape: array![2, 2].span(), data: array![1, 3, 5, 9].span(),
    );

    return tensor_cond.where(@tensor_1, @tensor_2);
}
>>> [1,4,5,8]
```
