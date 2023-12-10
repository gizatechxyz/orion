#tensor.less

```rust
    fn less(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
```

Check if each element of the first tensor is less than the corresponding element of the second tensor.
The input tensors must have either:
* Exactly the same shape
* The same number of dimensions and the length of each dimension is either a common length or 1.

## Args

* `self`(`@Tensor<T>`) - The first tensor to be compared
* `other`(`@Tensor<T>`) - The second tensor to be compared

## Panics

* Panics if the shapes are not equal or broadcastable

## Returns

A new `Tensor<usize>` of booleans (0 or 1) with the same shape as the broadcasted inputs.

## Examples

Case 1: Compare tensors with same shape

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn less_example() -> Tensor<usize> {
    let tensor_1 = TensorTrait::<u32>::new(
        shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    );

    let tensor_2 = TensorTrait::<u32>::new(
        shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 9, 1, 5].span(),
    );

    // We can call `less` function as follows.
    return tensor_1.less(@tensor_2);
}
>>> [0,0,0,0,0,0,1,0,0]
```

Case 2: Compare tensors with different shapes

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn less_example() -> Tensor<usize> {
    let tensor_1 = TensorTrait::<u32>::new(
        shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    );

    let tensor_2 = TensorTrait::<u32>::new(shape: array![3].span(), data: array![0, 1, 2].span(),);

    // We can call `less` function as follows.
    return tensor_1.less(@tensor_2);
}
>>> [0,0,0,0,0,0,0,1,1]
```
