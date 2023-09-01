#tensor.equal

```rust
    fn equal(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<usize>;
```

Check if two tensors are equal element-wise.
The input tensors must have either:
* Exactly the same shape
* The same number of dimensions and the length of each dimension is either a common length or 1.

## Args

* `self`(`@Tensor<T>`) - The first tensor to be equated
* `other`(`@Tensor<T>`) - The second tensor to be equated

## Panics

* Panics if the shapes are not equal or broadcastable

## Returns

A new `Tensor<usize>` of booleans (1 if equal, 0 otherwise) with the same shape as the broadcasted inputs.

## Examples

Case 1: Compare tensors with same shape

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, Tensor_u32_fp8x23};

fn eq_example() -> Tensor<usize> {
    let tensor_1 = TensorTrait::<u32>::new(
        shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    );

    let tensor_2 = TensorTrait::<u32>::new(
        shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 9, 1, 5].span(),
    );

    // We can call `equal` function as follows.
    return tensor_1.equal(@tensor_2);
}
>>> [1,1,1,1,1,0,0,0]
```

Case 2: Compare tensors with different shapes

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, Tensor_u32_fp8x23};

fn eq_example() -> Tensor<usize> {
    let tensor_1 = TensorTrait::<u32>::new(
        shape: array![3, 3, 3].span(), data: array![0, 1, 2, 3, 4, 5, 6, 7, 8].span(),
    );

    let tensor_2 = TensorTrait::<u32>::new(shape: array![3].span(), data: array![0, 1, 2].span(),);

    // We can call `equal` function as follows.
    return tensor_1.equal(@tensor_2);
}
>>> [1,1,1,0,0,0,0,0,0]
```
